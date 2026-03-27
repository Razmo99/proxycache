import { createOpencode } from "@opencode-ai/sdk";
import { readFile } from "node:fs/promises";
import path from "node:path";

function envInt(name, fallback) {
  const raw = process.env[name];
  return raw ? Number.parseInt(raw, 10) : fallback;
}

function buildLargePrompt(label, suffix, repeat) {
  const vocab = {
    atlas: ["orion", "nebula", "quasar", "pulsar", "cosmos", "galaxy", "meteor", "zenith"],
    zephyr: ["monsoon", "tempest", "gust", "squall", "cyclone", "breeze", "jetstream", "barometer"],
  };
  const words = vocab[label] ?? Array(8).fill(label);
  const blocks = Array.from({ length: repeat }, (_, index) =>
    `${label} ${words[0]} ${words[1]} ${words[2]} ${words[3]} ${words[4]} ${words[5]} ${words[6]} ${words[7]} marker${String(index).padStart(3, "0")}`,
  );
  return `${blocks.join("\n")}\n\nReply with exactly: ${suffix}`;
}

function normalizeMessages(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (Array.isArray(value?.data)) {
    return value.data;
  }
  return [];
}

function sessionUserMessages(messages) {
  return messages
    .filter((entry) => entry?.info?.role === "user")
    .map((entry) => {
      const text = entry.parts
        ?.filter((part) => part?.type === "text")
        .map((part) => part.text)
        .join("\n")
        .trim();
      return text ? { id: entry.info.id, content: text } : null;
    })
    .filter(Boolean);
}

async function readRepoFile(repoRoot, relativePath) {
  if (!repoRoot) {
    return "";
  }
  try {
    return await readFile(path.join(repoRoot, relativePath), "utf8");
  } catch {
    return "";
  }
}

async function buildLongSystemPrompt(repoRoot) {
  const agents = await readRepoFile(repoRoot, "AGENTS.md");
  const readme = await readRepoFile(repoRoot, "README.md");
  const fallback = [
    "You are OpenCode operating inside the proxycache repository.",
    "Prefer concise direct replies.",
    "Do not use tools unless explicitly required.",
    "Preserve repository conventions and follow the documented project guidance.",
  ].join("\n");
  const source = [agents, readme, fallback].filter(Boolean).join("\n\n");
  const sections = [
    "OpenCode default workspace context follows.",
    source || fallback,
  ];
  let prompt = sections.join("\n\n");
  while (prompt.length < 65536) {
    prompt += `\n\n[workspace context replay]\n${source || fallback}`;
  }
  return prompt;
}

function proxyMessagesForScenario(scenario, sessionMessages, longSystemPrompt) {
  const messages = sessionMessages.map((message) => ({ role: "user", content: message.content }));
  if (scenario === "long_context_revert" || scenario === "long_context_branch_thrash") {
    return [{ role: "system", content: longSystemPrompt }, ...messages];
  }
  return messages;
}

async function appendUserPrompt(client, sessionID, model, text) {
  await client.session.prompt({
    path: { id: sessionID },
    body: {
      agent: "smoke",
      noReply: true,
      model: { providerID: "llama.cpp", modelID: model },
      parts: [{ type: "text", text }],
    },
  });
}

async function currentSessionUsers(client, sessionID) {
  return sessionUserMessages(
    normalizeMessages(await client.session.messages({ path: { id: sessionID } })),
  );
}

function metricSnapshot(messages, proxyResult) {
  return {
    messageID: messages.at(-1)?.id ?? null,
    cacheRead: proxyResult.cacheRead,
    inputTokens: null,
    promptMs: proxyResult.promptMs,
  };
}

async function proxyChat(proxyUrl, model, messages) {
  const response = await fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer proxycache-real-smoke",
    },
    body: JSON.stringify({
      model,
      stream: false,
      temperature: 0,
      max_tokens: 24,
      messages,
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`proxy chat failed (${response.status}): ${JSON.stringify(payload)}`);
  }

  return {
    cacheRead: payload.timings?.cache_n ?? null,
    promptMs: payload.timings?.prompt_ms ?? null,
    content: payload.choices?.[0]?.message?.content ?? "",
  };
}

async function run() {
  const proxyUrl = process.env.OPENCODE_SMOKE_PROXY_URL;
  const model = process.env.OPENCODE_SMOKE_MODEL;
  const scenario = process.env.OPENCODE_SMOKE_SCENARIO ?? "basic";
  const repeat = envInt("OPENCODE_SMOKE_REPEAT", 96);
  const port = envInt("OPENCODE_SMOKE_PORT", 4096);
  const repoRoot = process.env.OPENCODE_SMOKE_REPO_ROOT;

  if (!proxyUrl || !model) {
    throw new Error("OPENCODE_SMOKE_PROXY_URL and OPENCODE_SMOKE_MODEL are required");
  }

  const longSystemPrompt =
    scenario === "long_context_revert" || scenario === "long_context_branch_thrash"
      ? await buildLongSystemPrompt(repoRoot)
      : "";

  const config = {
    "$schema": "https://opencode.ai/config.json",
    model: `llama.cpp/${model}`,
    small_model: `llama.cpp/${model}`,
    default_agent: "smoke",
    provider: {
      "llama.cpp": {
        npm: "@ai-sdk/openai-compatible",
        name: "llama-server (proxycache smoke)",
        models: {
          [model]: {
            name: `${model} (proxycache smoke)`,
          },
        },
        options: {
          apiKey: "proxycache-real-smoke",
          baseURL: `${proxyUrl}/v1`,
          timeout: 300000,
        },
      },
    },
    permission: {
      edit: "deny",
      bash: "deny",
    },
    agent: {
      smoke: {
        mode: "primary",
        model: `llama.cpp/${model}`,
        description: "Minimal direct-response agent for proxycache real smoke tests.",
        prompt:
          "You are a concise assistant used for smoke tests. Record the user's message only. Do not inspect the workspace, do not use tools, and do not create extra planning steps.",
        steps: 1,
        tools: {
          write: false,
          edit: false,
          bash: false,
        },
      },
    },
  };

  const opencode = await createOpencode({
    hostname: "127.0.0.1",
    port,
    timeout: 30000,
    config,
  });

  const { client } = opencode;
  const unwrap = (value) => value?.data ?? value;

  try {
    const session = unwrap(await client.session.create({
      body: { title: "Proxycache OpenCode SDK smoke" },
    }));

    const firstPrompt = scenario === "basic" ? buildLargePrompt("atlas", "SDK FIRST", repeat) : "Hi";
    await appendUserPrompt(client, session.id, model, firstPrompt);
    const firstMessages = await currentSessionUsers(client, session.id);
    const firstUser = firstMessages.at(-1);
    const firstProxy = await proxyChat(proxyUrl, model, proxyMessagesForScenario(scenario, firstMessages, longSystemPrompt));

    const secondPrompt = scenario === "basic" ? buildLargePrompt("zephyr", "SDK SECOND", repeat) : "Goodbye";
    await appendUserPrompt(client, session.id, model, secondPrompt);
    const secondMessages = await currentSessionUsers(client, session.id);
    const secondUser = secondMessages.at(-1);
    const secondProxy = await proxyChat(proxyUrl, model, proxyMessagesForScenario(scenario, secondMessages, longSystemPrompt));

    await client.session.revert({ path: { id: session.id }, body: { messageID: secondUser.id } });
    const revertedMessages = await currentSessionUsers(client, session.id);
    const afterRevertProxy = await proxyChat(proxyUrl, model, proxyMessagesForScenario(scenario, revertedMessages, longSystemPrompt));

    if (scenario === "long_context_branch_thrash") {
      await appendUserPrompt(client, session.id, model, "Summarize the repo in five words.");
      const branchOneMessages = await currentSessionUsers(client, session.id);
      const branchOneUser = branchOneMessages.at(-1);
      const branchOneProxy = await proxyChat(
        proxyUrl,
        model,
        proxyMessagesForScenario(scenario, branchOneMessages, longSystemPrompt),
      );

      await client.session.revert({ path: { id: session.id }, body: { messageID: branchOneUser.id } });
      const branchOneRevertedMessages = await currentSessionUsers(client, session.id);
      const afterSecondRevertProxy = await proxyChat(
        proxyUrl,
        model,
        proxyMessagesForScenario(scenario, branchOneRevertedMessages, longSystemPrompt),
      );

      await appendUserPrompt(client, session.id, model, "List one concrete proxy risk.");
      const finalBranchMessages = await currentSessionUsers(client, session.id);
      const finalBranchProxy = await proxyChat(
        proxyUrl,
        model,
        proxyMessagesForScenario(scenario, finalBranchMessages, longSystemPrompt),
      );

      const result = {
        scenario,
        sessionID: session.id,
        systemPromptChars: longSystemPrompt.length,
        revertedMessageID: secondUser.id,
        secondBranchRevertedMessageID: branchOneUser.id,
        first: metricSnapshot(firstMessages, firstProxy),
        second: metricSnapshot(secondMessages, secondProxy),
        afterRevert: metricSnapshot(branchOneRevertedMessages, afterRevertProxy),
        branchOne: metricSnapshot(branchOneMessages, branchOneProxy),
        afterSecondRevert: metricSnapshot(branchOneRevertedMessages, afterSecondRevertProxy),
        finalBranch: metricSnapshot(finalBranchMessages, finalBranchProxy),
        messageCount: branchOneRevertedMessages.length,
      };

      process.stdout.write(`${JSON.stringify(result)}\n`);
      return;
    }

    const result = {
      scenario,
      sessionID: session.id,
      systemPromptChars: longSystemPrompt.length,
      revertedMessageID: secondUser.id,
      revertState: null,
      first: metricSnapshot(firstMessages, firstProxy),
      second: metricSnapshot(secondMessages, secondProxy),
      afterRevert: metricSnapshot(revertedMessages, afterRevertProxy),
      messageCount: revertedMessages.length,
    };

    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally {
    opencode.server.close();
  }
}

run().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exit(1);
});
