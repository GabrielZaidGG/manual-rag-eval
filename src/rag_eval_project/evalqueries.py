EVAL_QUERIES = [
    {
        "query": "How safe are agents?",
        "type": "in_domain",
        "expect": "grounded_answer",
    },
    {
        "query": "Why do club deportivo guadalajara is the biggest LigaMx team?",
        "type": "out_of_domain",
        "expect": "refusal",
    },
    {
        "query": "What are the hidden risks when agents run unsupervised?",
        "type": "in_domain",
        "expect": "grounded_answer",  # retrieval fragile: "hidden risks" phrasing
    },
    {
        "query": "Explain how sandboxing vs containerization affects agent safety.",
        "type": "in_domain",
        "expect": "grounded_answer",  # fragile: overlapping technical terms
    },
    {
        "query": "Who won the last FIFA World Cup?",
        "type": "out_of_domain",
        "expect": "refusal",
    },
    {
        "query": "Can rogue agents leak secrets through side channels?",
        "type": "in_domain",
        "expect": "grounded_answer",  # fragile: “side channels” is niche
    },
    {
        "query": "Describe authentication layers in multi-agent orchestration.",
        "type": "in_domain",
        "expect": "grounded_answer",  # fragile: unusual phrasing “orchestration”
    },
    {
        "query": "Is Harry Potter based on real history?",
        "type": "out_of_domain",
        "expect": "refusal",
    },
    {
        "query": "How do agents resist adversarial prompts?",
        "type": "in_domain",
        "expect": "grounded_answer",  # fragile: “prompts” vs “inputs”
    },
    {
        "query": "What are the best tacos in Zapopan?",
        "type": "out_of_domain",
        "expect": "refusal",
    },
    {
        "query": "Why is continuous monitoring essential for agent compliance?",
        "type": "in_domain",
        "expect": "grounded_answer",  # fragile: compliance angle
    },
    {
        "query": "Explain isolation vs collaboration trade-offs for agent trust boundaries.",
        "type": "in_domain",
        "expect": "grounded_answer",  # fragile: “trust boundaries” phrasing
    },
    {
        "query": "Tell me about the latest Marvel movie release.",
        "type": "out_of_domain",
        "expect": "refusal",
    },
]
