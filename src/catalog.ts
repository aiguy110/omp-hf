/** One servable GGUF: the weights, an optional vision projector, and how to run them. */
export interface CatalogEntry {
	/** Model id under the `hf` provider; also the llama-server `--alias`. */
	id: string;
	name: string;
	repo: string;
	file: string;
	/** Vision projector in the same repo; enables image input. */
	mmproj?: string;
	/** Combined download size in bytes, for the picker. */
	bytes: number;
	contextWindow: number;
	maxTokens: number;
	reasoning: boolean;
	/** Chat template family; `qwen` enables the template's thinking controls. */
	family: "qwen" | "generic";
	/** Extra llama-server flags (recommended sampling, etc.). */
	serverArgs: string[];
	note?: string;
}

// Bonsai 2 small-model thinking profile (docs.prismml.com/run/server).
const BONSAI_SMALL_SAMPLING = [
	"--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
	"--presence-penalty", "1.0", "--repeat-penalty", "1.0", "--repeat-last-n", "64",
];
const BONSAI_27B_SAMPLING = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.05"];

function bonsai2(size: "2B" | "4B" | "9B" | "27B", bytes: number, contextWindow: number): CatalogEntry {
	return {
		id: `bonsai-2-${size.toLowerCase()}`,
		name: `Ternary Bonsai 2 ${size}`,
		repo: `prism-ml/Ternary-Bonsai-2-${size}-gguf`,
		file: `Ternary-Bonsai-2-${size}-PQ2_0.gguf`,
		mmproj: `Ternary-Bonsai-2-${size}-mmproj-Q8_0.gguf`,
		bytes,
		contextWindow,
		maxTokens: 16384,
		reasoning: true,
		family: "qwen",
		serverArgs: size === "27B" ? BONSAI_27B_SAMPLING : BONSAI_SMALL_SAMPLING,
		note: "PrismML ternary, PQ2_0",
	};
}

// Qwen3.5 thinking-mode sampling for general tasks (Qwen/Qwen3.5-*/README.md, "Best Practices").
function qwen35(size: "2B" | "4B", file: string, bytes: number): CatalogEntry {
	return {
		id: `qwen3.5-${size.toLowerCase()}`,
		name: `Qwen3.5 ${size}`,
		repo: `unsloth/Qwen3.5-${size}-GGUF`,
		file,
		mmproj: "mmproj-F16.gguf",
		bytes,
		contextWindow: 65536,
		maxTokens: 16384,
		reasoning: true,
		family: "qwen",
		serverArgs: ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0", "--presence-penalty", "1.5"],
		note: "Q4_K_M",
	};
}

// Gemma 4 thinking is toggled by a system-prompt token omp does not send, so these run non-thinking.
function gemma4(size: "E2B" | "E4B", bytes: number): CatalogEntry {
	return {
		id: `gemma-4-${size.toLowerCase()}-it`,
		name: `Gemma 4 ${size} IT`,
		repo: `ggml-org/gemma-4-${size}-it-GGUF`,
		file: `gemma-4-${size}-it-Q4_0.gguf`,
		mmproj: `mmproj-gemma-4-${size}-it-Q8_0.gguf`,
		bytes,
		contextWindow: 32768,
		maxTokens: 8192,
		reasoning: false,
		family: "generic",
		serverArgs: ["--temp", "1.0", "--top-p", "0.95", "--top-k", "64"],
		note: "Q4_0",
	};
}

export const CATALOG: readonly CatalogEntry[] = [
	bonsai2("2B", 517_312_992 + 364_664_352, 65536),
	bonsai2("4B", 1_146_465_120 + 366_894_624, 65536),
	bonsai2("9B", 2_416_851_840 + 624_229_664, 65536),
	bonsai2("27B", 7_206_168_928 + 629_246_976, 32768),
	qwen35("2B", "Qwen3.5-2B-Q4_K_M.gguf", 1_280_835_840 + 668_227_264),
	qwen35("4B", "Qwen3.5-4B-Q4_K_M.gguf", 2_740_937_888 + 672_423_616),
	gemma4("E2B", 2_841_481_184 + 557_368_064),
	gemma4("E4B", 4_590_807_392 + 559_874_816),
];

const QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q8_0"];

interface HfTreeItem {
	type: string;
	path: string;
	size: number;
}

/**
 * Build an entry for an arbitrary `user/repo[:QUANT]` reference by listing the
 * repo's GGUF files. Split (`-00001-of-0000N`) weights are not supported.
 */
export async function resolveHfRef(ref: string, signal?: AbortSignal): Promise<CatalogEntry> {
	const match = /^([\w.-]+\/[\w.-]+)(?::([\w.-]+))?$/.exec(ref.trim());
	if (!match) throw new Error(`Expected user/repo or user/repo:QUANT, got "${ref}"`);
	const [, repo, quant] = match;
	const res = await fetch(`https://huggingface.co/api/models/${repo}/tree/main`, { signal, headers: hfHeaders() });
	if (!res.ok) throw new Error(`Hugging Face returned ${res.status} for ${repo}`);
	const files = ((await res.json()) as HfTreeItem[]).filter(f => f.type === "file" && f.path.endsWith(".gguf"));
	const weights = files.filter(f => !/mmproj/i.test(f.path) && !/-\d{5}-of-\d{5}\.gguf$/.test(f.path));
	if (weights.length === 0) throw new Error(`${repo} has no single-file GGUF weights`);

	const byQuant = (q: string) => weights.find(f => f.path.toUpperCase().endsWith(`${q.toUpperCase()}.GGUF`));
	let pick = quant ? byQuant(quant) : undefined;
	if (quant && !pick) {
		throw new Error(`${repo} has no ${quant} file. Available: ${weights.map(f => f.path).join(", ")}`);
	}
	pick ??= QUANT_PREFERENCE.map(byQuant).find(Boolean) ?? weights.toSorted((a, b) => a.size - b.size)[0];

	const projectors = files.filter(f => /mmproj/i.test(f.path));
	const mmproj = projectors.find(f => /Q8_0/i.test(f.path)) ?? projectors.find(f => /f16/i.test(f.path)) ?? projectors[0];
	const base = pick.path.replace(/\.gguf$/i, "");
	return {
		id: base.toLowerCase().replace(/[^a-z0-9.-]+/g, "-"),
		name: base,
		repo,
		file: pick.path,
		mmproj: mmproj?.path,
		bytes: pick.size + (mmproj?.size ?? 0),
		contextWindow: 32768,
		maxTokens: 8192,
		reasoning: /qwen3|bonsai|deepseek-r1|thinking/i.test(repo),
		family: /qwen|bonsai/i.test(repo) ? "qwen" : "generic",
		serverArgs: [],
		note: "custom",
	};
}

export function hfHeaders(): Record<string, string> {
	const token = process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN;
	return token ? { Authorization: `Bearer ${token}` } : {};
}
