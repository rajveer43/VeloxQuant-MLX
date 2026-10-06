"""Corpus for the RAG-vs-KIVI benchmark.

Passages are grouped into four topics (KV cache internals, RAG mechanics,
MLX/Apple Silicon, quantization background) so TF-IDF retrieval has a real
discrimination task, and each passage is a full multi-paragraph piece (a
few hundred tokens) rather than a one-liner. This matters for the
benchmark's honesty: KIVI's memory case depends on the KV cache holding a
prefill long enough that its per-group quantization overhead is amortized
away, and a corpus of one-sentence passages caps every possible retrieved
context at a few hundred tokens no matter how many are retrieved. With
longer passages, retrieving most or all of the corpus (a high ``--k``)
produces a multi-thousand-token prompt, which is the regime KIVI's
reference benchmarks (and this repo's own long-context KIVI benchmarks)
actually validate against.

Each entry carries a topic tag purely for readability; the retriever does
not use it.
"""

from __future__ import annotations

CORPUS: list[dict[str, str]] = [
    # --- KV cache / inference systems ---
    {
        "topic": "kv_cache",
        "text": (
            "A key-value cache stores the attention keys and values computed for "
            "every previously generated token so a transformer does not need to "
            "recompute them on each new step. During the prefill phase, when the "
            "model processes the initial prompt, keys and values are computed for "
            "every token in that prompt and written into the cache in one pass. "
            "During the decode phase, each newly generated token only needs to "
            "compute its own key and value and then attend over everything stored "
            "so far, which is why generation is memory-bandwidth bound rather than "
            "compute bound once the cache is large. The cache's memory footprint "
            "grows linearly with sequence length, and separately scales with the "
            "number of transformer layers and the number of attention heads per "
            "layer, since every layer and every head keeps its own independent "
            "set of keys and values. For a model with many layers and a long "
            "context window, the KV cache can end up larger than the model's own "
            "weights once the sequence is long enough, which is why it has become "
            "a first-class target for compression research in its own right, "
            "separate from weight quantization."
        ),
    },
    {
        "topic": "kv_cache",
        "text": (
            "On Apple Silicon, model weights and the KV cache share the same "
            "unified memory pool as the operating system, the GPU driver, and "
            "every other running application. This makes KV cache size a binding "
            "constraint for long-context inference on Mac hardware in a way it "
            "usually is not on discrete-GPU systems, where the GPU has its own "
            "dedicated VRAM that is not shared with the host operating system. On "
            "a laptop with, say, 24 gigabytes of unified memory, a large language "
            "model's 4-bit weights might already consume several gigabytes before "
            "a single token of context is processed, and the KV cache then "
            "competes directly against that same budget as the context grows. "
            "This is part of why on-device long-context inference on consumer "
            "Apple Silicon hardware has historically been more memory-constrained "
            "than throughput-constrained: the model can technically generate "
            "tokens quickly enough, but there simply is not room in unified "
            "memory to hold the keys and values for a very long conversation or "
            "a very long retrieved document alongside everything else the "
            "system needs."
        ),
    },
    {
        "topic": "kv_cache",
        "text": (
            "KIVI is a tuning-free asymmetric quantization method for the KV "
            "cache, introduced in the paper 'KIVI: A Tuning-Free Asymmetric "
            "2bit Quantization for KV Cache' (Liu, Yuan et al., ICML 2024). Its "
            "central observation is that keys and values have different "
            "statistical structure and therefore want different quantization "
            "layouts. Keys tend to have a small number of high-variance "
            "channels that dominate attention scores, so KIVI quantizes keys "
            "per channel: it groups values along the token axis for each "
            "channel and computes one scale and zero point per group, which "
            "keeps those high-variance channels accurate. Values, by contrast, "
            "are flatter across channels but vary more from token to token, so "
            "KIVI quantizes values per token instead, grouping along the "
            "channel axis. On top of this asymmetric scheme, KIVI keeps the "
            "most recently generated tokens in full sixteen-bit precision "
            "inside what it calls a residual window, because recent tokens "
            "dominate attention scores and are cheap to store exactly; only "
            "once a token ages out of that window does it get quantized into "
            "the compressed region of the cache."
        ),
    },
    {
        "topic": "kv_cache",
        "text": (
            "Quantizing the KV cache trades a small amount of numerical "
            "accuracy for a large reduction in memory usage. A 2-bit KV cache "
            "can be roughly eight times smaller than a 16-bit fp16 cache for "
            "the tokens it actually compresses, since two bits per value "
            "versus sixteen bits per value is an eight-to-one reduction in raw "
            "storage. In practice the realized compression ratio is somewhat "
            "lower than that theoretical eight times, because group "
            "quantization schemes like KIVI's store a scale and a zero point "
            "per group of elements, and those scale and zero point values are "
            "themselves kept at higher precision, usually fp16, to avoid "
            "compounding quantization error. For a small group size, this "
            "per-group overhead is proportionally larger relative to the "
            "quantized codes themselves, which is why compression ratio "
            "generally improves as the quantized region of the cache grows "
            "large relative to the fixed number of scale-and-zero-point pairs "
            "needed to describe it."
        ),
    },
    {
        "topic": "kv_cache",
        "text": (
            "On Apple Silicon's Metal backend, KIVI's memory savings do not "
            "automatically translate into a throughput speedup, because the "
            "reference implementation's throughput advantage comes from a "
            "fused CUDA kernel that performs dequantization inline as part of "
            "the attention computation itself, avoiding a separate pass over "
            "the compressed cache. Metal has no direct equivalent of that "
            "fused kernel, so a Metal port of KIVI must dequantize the "
            "compressed keys and values back to full precision before handing "
            "them to the standard attention computation, which adds work "
            "rather than removing it. The expected win on Apple Silicon is "
            "therefore memory, not tokens per second: a smaller KV cache "
            "means more headroom in the shared unified memory pool for larger "
            "models or longer contexts, but it does not by itself make token "
            "generation faster, and in some configurations the extra "
            "dequantization work can make generation measurably slower than "
            "an uncompressed cache."
        ),
    },
    {
        "topic": "kv_cache",
        "text": (
            "The residual window is one of the more important practical "
            "design choices in KIVI, and in most group-quantized KV cache "
            "schemes generally. The idea is that the tokens generated most "
            "recently are, on average, the ones that matter most to the "
            "model's next prediction, both because of how attention scores "
            "tend to concentrate on nearby context and because errors "
            "introduced by quantizing those tokens would immediately and "
            "directly affect the very next token generated. By keeping the "
            "last R tokens, where R is a configurable residual length, in "
            "full sixteen-bit precision and only quantizing tokens once they "
            "fall outside that window, KIVI protects the part of the cache "
            "that is most sensitive to error while still compressing the "
            "much larger, older portion of the context that contributes less "
            "sharply to any individual next-token prediction. The tradeoff is "
            "that a larger residual window means more of the cache stays "
            "uncompressed, which reduces the realized compression ratio for "
            "short contexts where most of the sequence still falls inside "
            "the window."
        ),
    },
    # --- retrieval-augmented generation ---
    {
        "topic": "rag",
        "text": (
            "Retrieval-augmented generation, or RAG, combines a search step "
            "over a document collection with a language model's generation "
            "step, and was popularized as a way to let a language model "
            "answer questions using information it was never trained on, or "
            "information that has changed since training. The retrieved "
            "passages are inserted into the model's prompt, typically ahead "
            "of the user's actual question, so the model can ground its "
            "answer in specific source text rather than relying purely on "
            "facts encoded in its parameters during pretraining. This has two "
            "practical benefits: it reduces hallucination, because the model "
            "can quote or paraphrase from text that is directly in front of "
            "it rather than guessing from memory, and it lets a system stay "
            "current without retraining the underlying model, since updating "
            "the answer to a question is often as simple as updating the "
            "document collection the retriever searches over."
        ),
    },
    {
        "topic": "rag",
        "text": (
            "A typical RAG pipeline has two stages. First, a retriever ranks "
            "candidate passages from a document collection by their "
            "similarity to the user's query. This ranking step is usually "
            "implemented with either a dense method, such as a learned vector "
            "embedding compared by cosine similarity, or a sparse method, such "
            "as TF-IDF or BM25, which score documents by term overlap and "
            "term rarity rather than by a learned representation. Second, once "
            "the top-ranked passages are selected, a language model reads "
            "those passages together with the original query and produces a "
            "grounded answer. The choice between dense and sparse retrieval is "
            "itself a tradeoff: dense retrieval can capture semantic "
            "similarity that goes beyond exact word overlap, but it requires "
            "an embedding model and a vector index, while sparse retrieval "
            "needs no learned components at all and is fully deterministic "
            "and inspectable, which makes it attractive for small, "
            "self-contained systems where introducing a new dependency is "
            "undesirable."
        ),
    },
    {
        "topic": "rag",
        "text": (
            "Because RAG concatenates several retrieved passages into the "
            "prompt before the model even begins generating a response, the "
            "effective context length seen by the model grows directly with "
            "the number of retrieved passages and their individual lengths. "
            "This has a direct and often underappreciated consequence for "
            "system design: increasing the number of retrieved passages, "
            "often called k, to improve the odds that the relevant "
            "information is included in the context also increases the size "
            "of the KV cache the model must hold in memory during generation, "
            "since every token in every retrieved passage gets its own key "
            "and value stored across every layer and head of the model. A "
            "system that retrieves many long passages for every query is "
            "effectively choosing to run at a much longer context length than "
            "one that retrieves few short passages, and that choice has real "
            "memory and latency implications independent of how good the "
            "retrieval ranking itself is."
        ),
    },
    {
        "topic": "rag",
        "text": (
            "TF-IDF retrieval scores how relevant a document is to a query by "
            "combining two separate signals. Term frequency measures how "
            "often a query term appears within a specific document, on the "
            "reasoning that a document mentioning a term many times is more "
            "likely to be about that term. Inverse document frequency measures "
            "how rare that term is across the whole corpus, on the reasoning "
            "that a term appearing in nearly every document, such as a common "
            "word, carries little discriminating power, while a term that "
            "appears in only a handful of documents is far more informative "
            "about which specific documents are relevant. Multiplying these "
            "two signals together means that a document scores highly only "
            "when it contains query terms that are both frequent within that "
            "document and rare across the corpus as a whole, which in "
            "practice does a reasonable job of surfacing genuinely relevant "
            "documents without any learned model or training data."
        ),
    },
    {
        "topic": "rag",
        "text": (
            "Answer quality in a RAG system is often evaluated by checking "
            "whether the generated answer contains the expected facts or "
            "keywords from a gold reference answer, rather than requiring an "
            "exact string match, since language models rarely reproduce a "
            "reference answer word for word even when they get the underlying "
            "facts correct. Common approaches include keyword or phrase "
            "overlap scoring, where an answer is scored by what fraction of a "
            "predefined set of expected terms it mentions; automatic overlap "
            "metrics such as ROUGE, which compare n-gram overlap between the "
            "generated answer and a reference; and, increasingly, using "
            "another language model as a judge to assess whether an answer is "
            "factually consistent with the retrieved context. Each approach "
            "trades off cost, reproducibility, and nuance differently: keyword "
            "overlap is cheap, fully deterministic, and easy to audit by hand, "
            "but it can miss answers that are correct yet phrased very "
            "differently from the gold keywords."
        ),
    },
    {
        "topic": "rag",
        "text": (
            "The size of the document collection a retriever searches over "
            "does not need to be large for RAG to be useful; even a small, "
            "carefully curated corpus of a few dozen passages can meaningfully "
            "improve answer grounding compared to relying on a model's raw "
            "parametric knowledge, especially for niche or fast-changing "
            "topics. What matters more than raw corpus size is topical "
            "coverage and passage quality: a retriever can only surface "
            "information that actually exists somewhere in the collection, so "
            "a small corpus that thoroughly covers the topics a system expects "
            "to be asked about can outperform a much larger but noisier or "
            "more diffuse collection. This is part of why benchmark and "
            "evaluation corpora for RAG systems are often deliberately kept "
            "small and hand-curated rather than scraped indiscriminately from "
            "the web, since a small corpus makes it far easier to verify that "
            "every question in an evaluation set is actually answerable from "
            "the material the retriever has access to."
        ),
    },
    # --- Apple Silicon / MLX ---
    {
        "topic": "mlx",
        "text": (
            "MLX is an array framework designed for efficient machine "
            "learning research and inference on Apple Silicon. It borrows "
            "ideas from NumPy, PyTorch, and JAX, and its most distinctive "
            "design choice is unified memory: because Apple Silicon chips "
            "already share physical memory between the CPU and GPU at the "
            "hardware level, MLX arrays live in that same shared pool and can "
            "be operated on by either the CPU or the GPU without an explicit "
            "copy between separate memory spaces. MLX also uses lazy "
            "evaluation, meaning operations build up a computation graph "
            "rather than executing immediately, and the graph is only "
            "materialized when a result is actually needed, which lets MLX "
            "fuse and optimize sequences of operations before running them. "
            "This combination of unified memory and lazy evaluation is part "
            "of why MLX programs can be noticeably more memory-efficient on "
            "Apple Silicon than frameworks originally designed around "
            "discrete-GPU memory models."
        ),
    },
    {
        "topic": "mlx",
        "text": (
            "mlx_lm is a library built on top of MLX specifically for running "
            "large language models on Apple Silicon. It provides utilities "
            "for loading pretrained models, including models already "
            "quantized to four bits or lower, running text generation with "
            "configurable sampling parameters, and a pluggable KV cache "
            "interface that lets alternative cache implementations be "
            "substituted for the standard full-precision cache. This "
            "pluggable interface is what makes it possible to implement a "
            "custom compressed cache, such as a KIVI-style quantized cache, "
            "and use it as a drop-in replacement during generation: as long "
            "as the custom cache implements the same update-and-fetch "
            "protocol that mlx_lm's attention code expects, the rest of the "
            "generation pipeline, including sampling, tokenization, and the "
            "model's forward pass, does not need to change at all to benefit "
            "from the compressed cache."
        ),
    },
    {
        "topic": "mlx",
        "text": (
            "Metal is Apple's low-level graphics and compute API, and it is "
            "the layer through which MLX ultimately dispatches work to the "
            "GPU on Apple Silicon. Custom Metal kernels, written in a C-like "
            "shading language, can be compiled and dispatched directly from "
            "Python code to accelerate operations that are not well served "
            "by MLX's built-in operations, such as quantized group "
            "dequantization, where many small groups of values each need "
            "their own scale and zero point applied. Writing a dedicated "
            "Metal kernel for such an operation avoids a round trip through "
            "slower, more general-purpose code that would otherwise need "
            "several separate passes over the data, and it is a common "
            "pattern in KV-cache compression libraries to hand-write Metal "
            "kernels for the specific quantize and dequantize operations a "
            "given compression method needs, since those operations run on "
            "the hot path of every single generation step."
        ),
    },
    {
        "topic": "mlx",
        "text": (
            "Peak memory usage in an MLX program can be measured with a "
            "dedicated API that tracks the high-water mark of memory "
            "allocated on the GPU, and that peak counter can be reset between "
            "runs so that separate benchmark configurations do not "
            "contaminate each other's peak readings. This is important "
            "methodologically: without resetting the peak counter between, "
            "for example, an fp16 baseline run and a quantized-cache run, the "
            "reported peak for the second run could actually reflect memory "
            "allocated during the first run that was never freed, making the "
            "two configurations impossible to compare fairly. A correct "
            "benchmark resets the peak counter immediately before each "
            "configuration's generation call and reads it immediately after, "
            "so that the reported number reflects only the memory that "
            "configuration actually used."
        ),
    },
    {
        "topic": "mlx",
        "text": (
            "Apple Silicon chips use a unified memory architecture where the "
            "CPU, GPU, and Neural Engine all access the same physical pool of "
            "memory over a shared high-bandwidth interconnect, rather than "
            "each having its own dedicated memory as is typical on systems "
            "with a discrete GPU. This removes the need to explicitly copy "
            "data between separate host and device memory spaces, which on "
            "traditional GPU systems can be a meaningful source of latency "
            "for workloads that move data back and forth frequently. The "
            "tradeoff is that because the pool is shared, memory pressure "
            "from any one consumer, whether that is the operating system, "
            "another running application, or the model and its KV cache, "
            "directly reduces what is available to everything else, which is "
            "why memory efficiency work on Apple Silicon tends to focus as "
            "much on reducing peak usage as on reducing raw compute time."
        ),
    },
    # --- quantization background ---
    {
        "topic": "quantization",
        "text": (
            "Quantization reduces the number of bits used to represent a "
            "numeric value. A typical example converts a sixteen-bit "
            "floating point number into a low-bit-width integer code, such "
            "as two, three, or four bits, plus a small amount of side "
            "information, usually a shared scale and a zero point, that is "
            "used to reconstruct an approximation of the original value from "
            "the integer code. The core idea is that most numeric values in "
            "a neural network's weights or activations cluster within a "
            "relatively narrow range, so a small number of discrete levels "
            "can represent that range with acceptable error, while the "
            "memory savings from using far fewer bits per value can be "
            "substantial: a four-bit representation uses a quarter of the "
            "storage of a sixteen-bit one, and a two-bit representation uses "
            "an eighth, before accounting for the overhead of the scale and "
            "zero point themselves."
        ),
    },
    {
        "topic": "quantization",
        "text": (
            "Asymmetric quantization uses both a scale and a zero-point "
            "offset when mapping real values to integer codes, in contrast "
            "to symmetric quantization, which uses only a scale and assumes "
            "the value range is centered around zero. Asymmetric "
            "quantization can represent value ranges that are not centered "
            "at zero considerably more accurately, because it can shift the "
            "entire range to match wherever the actual data lies rather than "
            "wasting representable levels on values that never occur. This "
            "matters for KV cache quantization specifically because keys and "
            "values are not generally centered around zero and their "
            "distributions can shift meaningfully across different layers, "
            "different attention heads, and different channels within a "
            "head, so a scheme that can adapt its zero point locally, rather "
            "than assuming a fixed symmetric range everywhere, tends to "
            "produce noticeably lower quantization error in practice."
        ),
    },
    {
        "topic": "quantization",
        "text": (
            "Group quantization splits a tensor into small contiguous groups "
            "of elements and computes a separate scale and zero point for "
            "each group independently, rather than computing a single scale "
            "and zero point for the entire tensor at once. This better "
            "captures local variation in the data's range: if some regions "
            "of a tensor have a much wider spread of values than others, a "
            "single global scale would either waste precision on the "
            "narrow-range regions or clip and lose accuracy on the "
            "wide-range regions, whereas per-group scales let each region "
            "use a scale suited to its own local range. The tradeoff is "
            "storage overhead, since every group needs its own stored scale "
            "and zero point rather than sharing one across the whole tensor; "
            "smaller group sizes generally improve accuracy but increase "
            "that per-group overhead, while larger group sizes reduce "
            "overhead but can blur over local variation within a group."
        ),
    },
    {
        "topic": "quantization",
        "text": (
            "A quantization method is called tuning-free when it requires no "
            "calibration dataset or learned parameters to choose its scales, "
            "zero points, or codebooks, in contrast to methods that fit a "
            "codebook or set of quantization parameters to a representative "
            "sample of activations before deployment. Tuning-free methods "
            "compute their quantization parameters directly and "
            "deterministically from the data being quantized at the moment "
            "it is quantized, typically using simple statistics such as the "
            "minimum and maximum value within a group. This has two "
            "practical benefits: there is no separate calibration step or "
            "calibration dataset to maintain, and the method's output is "
            "fully deterministic and reproducible, since it involves no "
            "randomness or learned state that could vary between runs or "
            "between different calibration samples, which is a meaningful "
            "advantage for benchmarking and debugging."
        ),
    },
    {
        "topic": "quantization",
        "text": (
            "The cost of storing quantization side information, such as "
            "per-group scales and zero points, is a fixed overhead per group "
            "regardless of how large that group is, which means the "
            "proportional overhead of quantization depends heavily on how "
            "much data is actually being compressed relative to the number "
            "of groups needed to describe it. For a very small amount of "
            "data split into many small groups, the side information can "
            "approach or even exceed the size of the compressed codes "
            "themselves, largely erasing the benefit of quantizing at all. "
            "For a large amount of data, the same fixed per-group overhead "
            "is amortized across far more quantized values, so the realized "
            "compression ratio approaches the theoretical bits-per-value "
            "ratio much more closely. This is a general property of group "
            "quantization schemes, not specific to any one method, and it is "
            "why compression benchmarks that use very short sequences can "
            "understate or even invert the memory benefit that the same "
            "method shows on long sequences."
        ),
    },
]
