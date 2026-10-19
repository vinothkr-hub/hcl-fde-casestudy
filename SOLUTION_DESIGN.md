# Solution Design: Solstice Docs Q&A Assistant

## Problem and assumptions

Solstice support wants corporate admins to ask ordinary-language questions and
get answers from its product documentation. The key requirement is safe
abstention: when the docs do not support an answer, the assistant should say so
instead of guessing.

This v1 assumes the six supplied articles are the complete, trusted knowledge
base, and that returning a short verbatim excerpt with its article citation is
more useful and safer than generating a paraphrase. The five supplied questions
are the initial acceptance set, not a representative measure of all production
queries.

## Proposed approach

The implementation is a dependency-free Python command-line application. It
loads the JSON articles, tokenizes question and article text, removes common
question filler words, and normalizes simple English inflections. It ranks
whole articles using inverse-document-frequency-weighted query-term coverage.
This lightweight retrieval approach is adequate for the small dataset and has
no external service, credentials, or runtime cost.

An answer is returned only when the leading article matches at least two
distinct question terms and covers at least 35% of the question's weighted
terms. The assistant then returns the best-matching sentence from that article
verbatim and names its source. Otherwise it returns: “I don't have an answer
for that in the provided documentation.” The thresholds are intentionally
conservative starting points, not universal confidence probabilities.

The provided example set is the acceptance check: password reset and expiry,
CSV row limit, and Growth API rate limit should retrieve their expected
articles; the Salesforce question should abstain. The code and CLI use only the
Python standard library.

## Trade-offs

- **Retrieval instead of an LLM:** Exact excerpts are transparent and inherently
  grounded in the selected article. This avoids hallucinated additions,
  network dependencies, and model cost. The trade-off is less fluent output and
  weaker handling of synonyms or complex phrasing.
- **Whole-article ranking:** The dataset has only six short articles, so
  document-level search avoids unnecessary chunking complexity and cannot drop
  a short trailing chunk. A larger knowledge base would need sentence or
  passage-level indexing.
- **Explicit abstention gate:** Minimum term overlap plus weighted coverage
  handles the provided unsupported question. It is a simple heuristic, not a
  calibrated probability; it can reject valid paraphrases or accept a weak
  match on a future dataset. Thresholds should be tuned against labeled real
  questions and reviewed for false answers.
- **Verbatim sentence answers:** This preserves source wording and makes
  citations straightforward. A multi-part question may need more than one
  sentence, which this first version may not return.

## Next steps with more time or data

Collect representative admin questions, expected sources, and unsupported or
ambiguous examples; measure retrieval accuracy and unsafe-answer/abstention
rates; then tune the gate. For broader documentation, add passage-level
retrieval and metadata filters. If customers need conversationally phrased
answers, add an optional LLM only after retrieval, constrain it to cited
passages, and verify it does not introduce claims absent from those passages.
For deployment, add a web interface, observability for unanswered questions,
and an editorial workflow for keeping the indexed docs current.
