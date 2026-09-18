# Prompts used in the benchmark (verbatim)

## GENERATION_2CLASS

```text
Task: scientific claim verification using ONLY the evidence sentences provided.

Output contract:
- Return ONLY a valid JSON object, no markdown, no extra text.
- Keys: "label", "answer", "evidence".
- "label" must be exactly one of: SUPPORTS, REFUTES.
- "answer": one or two concise sentences grounded in the evidence.
- "evidence": list of evidence ids (strings copied verbatim from the brackets) that justify the label. Use [] if none apply.

Claim:
{claim}

Evidence sentences:
{evidence}

JSON:
```

## GENERATION_3CLASS

```text
Task: scientific claim verification using ONLY the evidence sentences provided.

Output contract:
- Return ONLY a valid JSON object, no markdown, no extra text.
- Keys: "label", "answer", "evidence".
- "label" must be exactly one of: SUPPORTS, REFUTES, NOT_ENOUGH_INFO.
  Use NOT_ENOUGH_INFO when the evidence neither supports nor refutes the claim.
- "answer": one or two concise sentences grounded in the evidence.
- "evidence": list of evidence ids (strings copied verbatim from the brackets) that justify the label. Use [] if none apply.

Claim:
{claim}

Evidence sentences:
{evidence}

JSON:
```

## CRAG_EVALUATOR

```text
You are a retrieval evaluator for scientific claim verification.
For each retrieved sentence, rate how relevant it is for deciding whether the claim is supported or refuted.
Score 1.0 = the sentence directly addresses the claim; 0.0 = unrelated.

Return ONLY a JSON object of the form {{"scores": {{"<id>": <score between 0 and 1>, ...}}}} containing every id.

Claim:
{claim}

Retrieved sentences:
{evidence}

JSON:
```

## QUERY_REFORMULATION

```text
Rewrite the following scientific claim as a search query for a corpus of biomedical abstracts.
Keep all technical entities, add common synonyms or expanded abbreviations, and remove hedging words.
Return ONLY a JSON object: {{"query": "<rewritten query>"}}.

Claim:
{claim}

JSON:
```

## AGENT_CONTROLLER

```text
You are the controller of an iterative retrieval agent that gathers evidence to verify a scientific claim.
You can act at most {remaining} more time(s) before the answer must be produced.

Decide the next action:
- {{"action": "search", "query": "<new search query>", "thought": "<why>"}}  - issue another search with a reformulated or decomposed query
  (for example a sub-question, a synonym, an expanded abbreviation, or a focus on a specific entity).
- {{"action": "finish", "thought": "<why>"}}  - the collected evidence is sufficient (or nothing more can be found).

Return ONLY the JSON object.

Claim:
{claim}

Search history:
{history}

Evidence collected so far (id: sentence):
{evidence}

JSON:
```

## ENTITY_RELATION_EXTRACTION

```text
Extract a knowledge graph from the scientific abstract below.

Return ONLY a JSON object with two keys:
- "entities": list of objects {{"name": "<canonical entity name>", "type": "<one of: disease, gene_or_protein, drug_or_chemical, cell_or_tissue, organism, method, measure, concept, other>"}}
- "relations": list of objects {{"source": "<entity name>", "target": "<entity name>", "relation": "<short verb phrase>", "sentence_ids": [<ids of sentences that express the relation>]}}

Guidelines: use short canonical names (expand abbreviations once, e.g. "interleukin-6"), include the 5-15 most important entities,
and only list relations that are explicitly stated. Sentence ids are the integers in brackets.

Title: {title}

Abstract sentences:
{sentences}

JSON:
```

## LIGHTRAG_KEYWORDS

```text
Extract retrieval keywords from the scientific claim below at two levels.

Return ONLY a JSON object:
{{"low_level": ["<specific entities: genes, proteins, drugs, diseases, cell types, methods, measured quantities>"],
  "high_level": ["<broader themes or relation types: e.g. 'risk factor', 'gene expression regulation', 'treatment efficacy'>"]}}

Claim:
{claim}

JSON:
```

## COMMUNITY_SUMMARY

```text
Summarise the following group of related entities and relations from a corpus of scientific abstracts
in 3-5 sentences, focusing on the shared theme and the main findings. Return plain text.

Entities:
{entities}

Relations:
{relations}

Summary:
```

