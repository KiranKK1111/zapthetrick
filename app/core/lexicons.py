"""Central lexicon registry — ALL deterministic decision-path terminology.

Every hand-written regex pattern / keyword list that GATES behavior (classify
an intent, pick a route, detect a slot, veto an escalation) lives HERE as pure
data, so the terminology surface of the codebase is auditable and tunable in
one file (see AnalysisReports/Status.md — the hardcoded-terminology audit).

Contract:
  * This module holds DATA ONLY — raw pattern strings and phrase tuples. No
    imports from app code, no compiled regexes, no logic. Consumers compile
    (`re.compile(lexicons.X, re.IGNORECASE)`) and keep their local names, so
    call sites and tests are unchanged.
  * Names are prefixed by consumer domain: INTENT_* (clarify/intent_pipeline),
    DIFFICULTY_* (chat/difficulty), ACT_* (followup/acts), QD_*
    (question_detection/classifier), TRIAGE_* (chat/triage), LIVE_* (live/*).
  * These are the deterministic FLOOR (Architecture §9): fail-open fast paths
    with an LLM/semantic backstop. Safety/PII/injection patterns deliberately
    stay in their own modules (they must not gain an import surface).
"""
from __future__ import annotations

# =========================================================================
# clarify/intent_pipeline.py — intent classification + slot extraction
# =========================================================================

INTENT_GREETING = (
    r"^\s*(hi+|hey+|hello+|yo|sup|howdy|gm|good (morning|afternoon|evening|"
    r"night)|thanks?|thank you|thx|ty|ok(ay)?|cool|nice|great|got it|bye|"
    r"goodbye|cheers)\b[\s!.]*$"
)
INTENT_KNOWLEDGE = (
    r"\b(explain|describe|what(?:'s| is| are| does)?|how (?:do|does|to|can)|"
    r"why|tell me about|summari[sz]e|overview|understand|meaning of|"
    r"define|definition)\b"
)
# Read-only "explain EXISTING content" request (explanation verb at the HEAD +
# a reference to already-present code/text) — never clarify these.
INTENT_EXPLAIN_EXISTING = (
    r"^\s*(?:(?:can|could|would|will|pls|please|kindly|hey|could\s+u|can\s+u)\s+)*"
    r"(?:you|u|someone|somebody)?\s*"
    r"(explain|describe|review|analy[sz]e|summari[sz]e|interpret|clarify|"
    r"walk\s+me\s+through|break\s*down|go\s+through|understand|"
    r"what(?:'s| is| does| are| do)|how\s+(?:does|do|is|are))\b"
    r"[^.?!\n]{0,60}?\b(this|that|these|it|above|below|following|"
    r"the\s+(?:code|program|function|method|snippet|script|class|file|logic|"
    r"algorithm|implementation|output|error|query|command|snippet))\b"
)
# Signals that the message contains PASTED code.
INTENT_CODE_PASTE = (
    r"```|\b(def |class |public |private |protected |static |void |import |"
    r"#include|function\s+\w+\s*\(|=>|console\.log|System\.out|printf?\(|"
    r"return\s|const |let\s+\w+\s*=|var\s+\w+\s*=|for\s*\(|while\s*\()"
)
INTENT_COMPARISON = (
    r"\b(compare|comparison|vs\.?|versus|difference between|better|"
    r"pros and cons|trade[- ]?offs?)\b"
)
INTENT_DEBUG = (
    r"\b(fix|debug|error|exception|stack ?trace|not working|doesn'?t work|"
    r"broken|bug|why (is|does).*(fail|crash|throw)|traceback)\b"
)
INTENT_TEST = (
    r"\b(unit ?tests?|write tests?|test cases?|pytest|junit|jest|"
    r"test (this|the|my)|add tests?)\b"
)
INTENT_DOCS = (
    r"\b(document(ation)?|readme|docstrings?|api ?spec|openapi|swagger|"
    r"comment(s| this)?|javadoc)\b"
)
INTENT_DESIGN = (
    r"\b(architect(ure)?|system design|design (a|the|an)|high[- ]?level "
    r"design|data ?model|schema design|diagram|er ?diagram)\b"
)
INTENT_CODE_GEN = (
    r"\b(write|implement|code|program|function|method|snippet|algorithm|"
    r"give me|need|want|generate|show me)\b"
)
# Concrete, self-contained operations → directly answerable code task.
INTENT_OPERATION = (
    r"\b(reverse|sort|search|find|parse|validate|convert|transform|calculate|"
    r"compute|count|sum|average|merge|filter|map|reduce|format|encode|decode|"
    r"encrypt|decrypt|hash|compress|serialize|deserialize|fibonacci|factorial|"
    r"palindrome|prime|gcd|lcm|fizzbuzz|swap|rotate|flatten|deduplicate|"
    r"traverse|insert|delete|update|append|split|join|replace|match|"
    r"tokenize|shuffle|binary search|quick ?sort|merge ?sort|dijkstra|"
    r"regex|read|write|upload|download|authenticate|paginate)\b"
)
INTENT_PLATFORM = (
    r"\b(web ?app|website|web|browser|mobile|android|ios|desktop|"
    r"windows|macos|linux|cli|command[- ]?line|terminal|backend|frontend|"
    r"server[- ]?side|api)\b"
)
INTENT_FRAMEWORK = (
    r"\b(react|angular|vue|svelte|next\.?js|nuxt|django|flask|fastapi|"
    r"spring|spring ?boot|rails|laravel|symfony|nestjs|express|flutter|"
    r"swiftui|jetpack|xamarin|ionic|electron|streamlit|qt|unity|godot)\b"
)
# Technique/constraint cues that raise specificity (→ higher confidence).
INTENT_CONSTRAINT = (
    r"\b(streams?|recursion|recursive|iterative|async|await|threads?|"
    r"concurren\w+|in[- ]?place|one[- ]?liner|regex|generics?|immutable|"
    r"functional|lambda|stack|queue|without using \w+|using \w+)\b"
)
# Vague scope markers that lower confidence / raise ambiguity.
INTENT_VAGUE = (
    r"\b(something|some kind of|stuff|things?|whatever|anything|a thing|"
    r"etc\.?|and so on|like that)\b"
)
INTENT_DOC_FORMAT = (
    r"\b(pdf|docx?|word|markdown|md|readme|read me|excel|xlsx|csv|pptx|"
    r"powerpoint|slides?|html|inline comments?|code comments?|javadoc|"
    r"docstrings?|api ?docs?|api ?spec|openapi|swagger)\b"
)
INTENT_ARCHIVE_VERB = (
    r"\b(compress(?:ed)?|archive[ds]?|zip(?:\s*it(?:\s*up)?| up)?|tarball|"
    r"package[ds]?|packaging|bundle[ds]?|bundling|download|export)\b"
)
# A bare-noun archive request that needs no explicit target ("get me the
# archive", "i want the archive", "can I get an archive", "the project
# archive", "as a single/one file/bundle"). These phrasings mean "package the
# thing we've been working on" even without a project/code target word — so
# archive intent isn't tied to any one keyword like "download".
INTENT_ARCHIVE_NOUN = (
    r"(?:\b(?:the|an?|get|want|need|send|give)\b[\w\s]{0,20}?\barchive\b"
    r"|\barchive\s+of\b"
    r"|\bas\s+(?:a\s+)?(?:single|one|1)\s+(?:file|archive|bundle|package)\b"
    r"|\bas\s+(?:a\s+)?(?:zip|archive|bundle|package|tarball)\b)"
)
INTENT_ARCHIVE_FORMAT = (
    r"\b(zip|tar\.gz|tar\.bz2|tar\.xz|tgz|tar|7z|7-?zip|rar|gzip|gz|bz2|xz)\b"
)
# The ONLY archive formats we can create — anything else → ask.
INTENT_SUPPORTED_ARCHIVE = r"\b(zip|\.zip|7z|7-?zip|sevenz)\b"

# Named document GENRES: a request naming one ("API design document", "PRD",
# "test plan") has already said WHAT to produce — the format has a safe default
# (markdown/pdf), so asking "which format?" would be an unnecessary
# clarification (decision matrix: "user already specified document type").
INTENT_DOC_GENRE = (
    r"\b(design document|api design|architecture document|prd|product "
    r"requirements|spec(?:ification)?s?|requirements? (?:document|doc)|"
    r"proposal|report|user stor(?:y|ies)|test plan|rfc|adr|runbook|"
    r"post[- ]?mortem|white ?paper|case study|release notes|changelog|"
    r"style guide|onboarding (?:guide|doc)|readme|license|"
    r"contributing (?:guide|doc))\b"
)
# Non-code deliverables a generate/write/create verb can produce — these must
# NOT trigger the code-generation "which language?" ask (acceptance criteria
# aren't code). With a subject ("for login") they're directly answerable;
# without one, the missing piece is the SUBJECT, not a language.
INTENT_NONCODE_DELIVERABLE = (
    r"\b(requirements?|acceptance criteria|user stor(?:y|ies)|test plan|"
    r"test strategy|uml|class diagram|sequence diagram|activity diagram|"
    r"use[- ]?case diagram|er[- ]?diagram|flow ?chart|prd|"
    r"spec(?:ification)?s?|documentation|design document|architecture "
    r"diagram|wireframes?|mock-?ups?|roadmap|estimates?|sprint plan|"
    # SeveralFeatures.md scenario families: writing/office deliverables and
    # data seeds are NEVER a "which programming language?" ask.
    r"presentations?|slide ?decks?|slides|powerpoint|e-?mails?|memos?|"
    r"(?:cover )?letters?|blog posts?|articles?|essays?|"
    r"(?:mock|sample|seed|dummy) data)\b"
)
# Reference to the USER'S OWN artifact (their code / screenshot / logs / …) —
# an analyze-type ask about one of these with NOTHING attached/pasted means
# the required input is missing ("Fix my code" → "please paste the code").
INTENT_ARTIFACT_REF = (
    r"\b(?:my|our|this|these|that|the)\s+"
    r"(?:code(?:base)?|program|script|function|method|class|snippet|"
    r"app(?:lication)?|api|service|website|project|repo(?:sitory)?|"
    r"architecture|diagram|screenshot|image|photo|logs?|error|exception|"
    r"stack ?trace|traceback|crash|query|schema|database|dependenc(?:y|ies)|"
    r"config(?:uration)?|file|test suite)\b"
)
# Verbs that operate ON an existing artifact (vs producing something new).
INTENT_ANALYZE_VERB = (
    r"\b(fix|debug|explain|review|analy[sz]e|refactor|optimi[sz]e|improve|"
    r"audit|check|inspect|trace|profile|walk me through|understand|"
    r"summari[sz]e|document|test|secure|modify|update|change|edit|extend)\b"
)
# Explicit QUANTITATIVE constraints in a request ("at least 500 ms", "under
# 2 seconds", "max 100 MB") — extracted verbatim into a hard directive so the
# model can't invert them (the classic failure: "at least 500 ms" answered
# with the fastest solution and a claim it's "within 500 ms").
INTENT_QUANT_CONSTRAINT = (
    r"\b(at ?least|atleast|minimum(?: of)?|no (?:less|fewer) than|more than|"
    r"over|at ?most|maximum(?: of)?|no more than|less than|under|within|"
    r"between|exactly|around|roughly|approximately)\b"
    r"[^.\n]{0,40}?\b\d[\d,.]*\s*"
    r"(ms|msec|millisecond(?:s)?|s\b|sec(?:ond)?s?|minute(?:s)?|"
    r"mb|gb|kb|bytes?|lines?|files?|characters?|words?|items?|elements?)\b"
)
# Deliberately-suboptimal asks ("brute force", "naive", "slower", "without
# optimization") — the user WANTS a less-optimal technique; the model must not
# helpfully substitute the fastest one.
INTENT_SUBOPTIMAL_ASK = (
    r"\b(brute[- ]?force|naive|naïve|unoptimi[sz]ed|less optimal|"
    r"sub[- ]?optimal|slower|inefficient|simplest possible|"
    r"without (?:any )?optimi[sz]ations?|deliberately slow)\b"
)

# Imperative starters for the VAGUE-TASK rule: a very short command with no
# tech, no artifact, and no concrete operation is under-specified ("Deploy my
# application.", "Design a database.", "Add monitoring."). Explanation verbs
# (explain/describe/what-is) are deliberately absent — a short knowledge
# question is answerable, not vague.
INTENT_VAGUE_IMPERATIVE = (
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(deploy|migrate|modernize|upgrade|optimi[sz]e|improve|review|"
    r"analy[sz]e|estimate|prepare|add|create|build|design|generate|write|"
    r"make|develop|perform|conduct|set ?up|implement|refactor|fix|debug|"
    r"test|secure|scale|monitor|integrate|automate|containeri[sz]e|"
    r"apply|use|investigate|compare|evaluate|assess|benchmark|modify|"
    r"port|convert|which|what)\b"
)

# =========================================================================
# chat/difficulty.py — fast-path difficulty routing
# =========================================================================

# Greetings / acknowledgements that are obviously trivial (skip the LLM).
DIFFICULTY_TRIVIAL_PHRASES = frozenset({
    "hi", "hii", "hey", "hello", "yo", "sup", "hiya", "howdy", "gm",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thx", "ty", "ok", "okay", "kk", "cool", "nice",
    "great", "awesome", "got it", "bye", "goodbye", "see ya", "cheers",
})
# Heavy/large-scope generation signals → route straight to the strongest model.
DIFFICULTY_HEAVY = (
    r"\b("
    r"whole|entire|complete|full|end[- ]?to[- ]?end|production[- ]?(?:ready|grade)|"
    r"enterprise|monorepo|micro[- ]?services?|large[- ]?scale|full[- ]?stack"
    r")\b.*\b("
    r"project|app|application|system|platform|codebase|website|backend|frontend|"
    r"solution|product|repo|repository|service"
    r")\b"
)
# Explicit magnitude: "1000 files", "100000 lines", "50 components…".
DIFFICULTY_MAGNITUDE = (
    r"\b(\d{3,})\s*(files?|components?|modules?|endpoints?|pages?|screens?|"
    r"services?|microservices?|tables?)\b"
    r"|\b(\d{4,})\s*(lines?|loc)\b"
)
# Explanation/overview cues — veto the heavy-generation escalation.
DIFFICULTY_EXPLAIN = (
    r"\b(explain|describe|summari[sz]e|summary|overview|walk ?through|"
    r"what(?:'s| is| are| does)|how (?:do|does|to)|why|tell me about|"
    r"understand|review|analyse|analyze|compare|difference)\b"
)
# Open-ended PROJECT nouns (multi-file deliverables; excludes program/script).
DIFFICULTY_PROJECT_NOUN = (
    r"\b(project|app|apps|application|web ?app|website|web ?site|api|service|"
    r"system|dashboard|game|backend|frontend|full[- ]?stack|platform|bot|"
    r"extension|plugin|clone|saas|crm|marketplace|e-?commerce|"
    r"micro[- ]?services?)\b"
)
# Verbs that genuinely start a project build.
DIFFICULTY_PROJECT_VERB = r"\b(build|create|make|generate|develop|scaffold)\b"
# Language/framework/tech names — "tech named?" ambiguity gate for builds AND
# the "user already specified their stack" suppression (decision matrix: never
# ask about a language/framework/database/cloud the user already named).
DIFFICULTY_TECH = (
    r"(?<!\w)(python|javascript|js|typescript|ts|java|kotlin|swift|scala|dart|"
    r"go|golang|rust|ruby|php|perl|lua|haskell|elixir|erlang|clojure|julia|"
    r"matlab|solidity|objective-?c|c#|csharp|c\+\+|cpp|sql|bash|shell|"
    r"powershell|react|angular|vue|svelte|node|nodejs|django|flask|fastapi|"
    r"spring|spring ?boot|rails|laravel|symfony|nestjs|express|dotnet|\.net|"
    r"next\.?js|nuxt|flutter|swiftui|jetpack|compose|xamarin|ionic|cordova|"
    r"html|css|tailwind|bootstrap|android|ios|qt|electron|streamlit|unity|"
    r"godot|pandas|numpy|pytorch|tensorflow|pygame|"
    # databases / stores
    r"postgres(?:ql)?|mysql|mariadb|mongodb|mongo|redis|sqlite|oracle ?db|"
    r"dynamodb|cassandra|elasticsearch|opensearch|couchdb|neo4j|snowflake|"
    r"bigquery|clickhouse|supabase|firebase|firestore|"
    # cloud / devops / infra
    r"aws|azure|gcp|google cloud|kubernetes|k8s|docker|terraform|"
    r"cloudformation|pulumi|ansible|helm|eks|gke|aks|lambda|ec2|s3|fargate|"
    r"cloudflare|vercel|netlify|heroku|github actions?|gitlab ci|jenkins|"
    r"circleci|argo ?cd|prometheus|grafana|datadog|"
    # messaging / streaming
    r"kafka|rabbitmq|pulsar|sqs|sns|nats|zeromq|celery|"
    # API styles / protocols (concrete choices, unlike bare 'rest')
    r"graphql|grpc|websockets?|openapi|"
    # AI / LLM stacks
    r"llama|gpt-?[0-9a-z]*|openai|anthropic|claude|gemini|mistral|deepseek|"
    r"rag|langchain|llamaindex|hugging ?face|transformers|bert|whisper|"
    r"ollama|vllm|pgvector|qdrant|pinecone|weaviate|chroma|"
    # test frameworks (naming one = the tooling choice is made)
    r"junit|pytest|jest|mocha|cypress|selenium|playwright|testng|rspec)(?!\w)"
)

# =========================================================================
# chat/triage.py — document-intent precision gate
# =========================================================================

# A downloadable-artifact token: the precision guard that lets us trust the
# LLM's `document:true` under explicit-only mode.
TRIAGE_ARTIFACT = (
    r"\b(pdf|docx?|word|excel|xlsx|spreadsheet|powerpoint|pptx?|ppt|slide|"
    r"slides|slide\s*deck|presentation|csv|json|markdown|\.md|txt|text\s+file|"
    r"zip|archive|tarball|document|file|attachment|download|export|"
    r"downloadable)\b"
)

# =========================================================================
# followup/acts.py — conversational-act lexicons
# =========================================================================

ACT_APPROVAL = (
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "sounds good",
    "looks good", "perfect", "great", "do it", "go ahead", "lgtm", "approved",
    "that works", "works for me", "agreed", "confirm", "confirmed",
)
ACT_REJECTION = (
    "no", "nope", "nah", "wrong", "incorrect", "that's not", "thats not",
    "not what i", "don't", "do not", "stop that", "cancel that",
)
ACT_CONTINUATION = (
    "continue", "go on", "keep going", "carry on", "proceed", "the rest",
    "and then", "finish it", "finish the", "what's next", "whats next",
)
ACT_CORRECTION = (
    "actually", "i meant", "i mean", "instead", "rather", "correction",
    "change it to", "should be", "not that", "undo", "revert", "never mind",
    "nevermind", "scratch that", "remove ", "no longer",
)
ACT_COMPARISON = (
    "compare", " vs ", " versus ", "difference between", "which is better",
    "which one", "pros and cons", "trade-off", "tradeoff", "better than",
)
ACT_EXPANSION = (
    "explain more", "elaborate", "expand", "in more detail", "in detail",
    "go deeper", "tell me more", "more detail", "more about", "why exactly",
)
ACT_IMPROVE = (
    "make it better", "improve", "optimize", "make it faster", "faster",
    "cleaner", "refine", "polish", "enhance", "better",
)
# Explicit TOPIC-SHIFT cues — must win over correction/continuity lexicons.
ACT_TOPIC_SHIFT = (
    "new topic", "different topic", "another topic", "change of topic",
    "change the topic", "change of subject", "change the subject",
    "changing subject", "changing the subject", "different subject",
    "different question", "another question", "separate question",
    "unrelated question", "unrelated to", "on a different note",
    "on another note", "switching gears", "switch gears", "let's switch",
    "lets switch", "let's move on", "lets move on", "moving on",
    "let's move to", "lets move to", "something else", "different thing",
    "forget that", "forget about that", "forget the previous", "forget what",
    "let's talk about something", "lets talk about something",
    "let's change", "lets change", "put that aside", "set that aside",
    "leaving that aside",
)
# Reference cues — pronouns + selection references (follow-up confidence).
ACT_PRONOUNS = (
    "it", "that", "this", "those", "these", "them", "they", "same", "the above",
    "the previous", "the last one", "above",
)
ACT_SELECTION = (
    r"\b(the\s+)?(first|second|third|fourth|fifth|last|next|previous|"
    r"\d+(st|nd|rd|th))\b|\boption\s+[a-z0-9]\b"
)

# =========================================================================
# question_detection/classifier.py — fast-path question hints (LLM-backed)
# =========================================================================

QD_INTERROGATIVES = (
    "what", "how", "why", "when", "where", "who", "which", "whose",
    "tell", "walk", "describe", "explain", "compare", "discuss",
    "can you", "could you", "would you", "have you", "did you", "do you",
    "why don't", "what's", "how's", "give me",
)
QD_FOLLOWUP_STARTERS = (
    # A question that OPENS on a conjunction / back-reference continues the
    # prior thread ("And why is that?", "So how does that scale?", "But what
    # about failures?"). Deterministic follow-up signal for the heuristic
    # path (the LLM classifier still does the richer, context-aware call).
    "and ", "and,", "so ", "but ", "also ", "okay and", "ok and",
    "what about", "how about", "why is that", "why that", "what else",
    "and why", "and how", "and what", "then how", "then what", "then why",
)
QD_CODING_HINTS = (
    "implement", "write a function", "code", "algorithm", "complexity",
    "leetcode", "data structure", "reverse a", "sort", "binary tree",
    "linked list", "array", "string", "hash map", "recursion", "dp",
    "dynamic programming",
)
QD_BEHAVIORAL_HINTS = (
    "tell me about a time", "describe a situation", "how do you handle",
    "conflict", "teamwork", "leadership", "weakness", "strength",
    "why should we hire", "challenge you faced",
)
QD_SMALLTALK_HINTS = (
    "how are you", "nice to meet", "good morning", "good afternoon",
    "thanks for", "thank you for",
)

# =========================================================================
# live/* — interview copilot cue lexicons
# =========================================================================

# --- live/coach.py — candidate delivery coaching -------------------------
LIVE_COACH_FILLERS = ("um", "uh", "like", "you know", "basically", "actually", "sort of",
            "kind of", "i mean", "right")
LIVE_COACH_EXAMPLE_CUES = ("for example", "for instance", "in my project", "at my last",
                 "we built", "i implemented", "such as", "e.g")

# --- live/diarize.py — speaker role / hand-off cues ----------------------
# Textual cues that hint at a role/hand-off within an interviewer turn.
LIVE_DIARIZE_HANDOFF_CUES = ("my colleague", "hand over to", "i'll let", "over to you",
                 "pass it to", "my co-interviewer", "another question from")
LIVE_DIARIZE_RECRUITER_CUES = ("salary", "compensation", "notice period", "next steps",
                   "scheduling", "availability", "hr ")
LIVE_DIARIZE_HIRING_MGR_CUES = ("the team", "on my team", "you'd report to", "our roadmap",
                    "headcount")

# --- live/events.py — utterance-event typing cues ------------------------
LIVE_EVENTS_INTERROGATIVE = (
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "can", "could", "would", "should", "do", "does", "did", "is", "are",
    "was", "were", "will", "have", "has", "explain", "describe", "tell",
    "walk", "compare", "discuss", "give", "define", "name",
)
LIVE_EVENTS_GREETING_CUES = ("hello", "hi ", "hey", "good morning", "good afternoon",
                  "good evening", "nice to meet", "how are you")
LIVE_EVENTS_ACK_CUES = ("okay", "ok ", "got it", "makes sense", "thank you", "thanks",
             "great", "perfect", "cool", "alright", "right.", "good.")
LIVE_EVENTS_TRANSITION_CUES = ("let's move", "lets move", "moving on", "next", "let's talk",
                    "lets talk", "let's discuss", "lets discuss", "now let")

# --- live/implicit.py — implicit-question detection ----------------------
# Imperative / probing cues that signal an implicit request to respond.
LIVE_IMPLICIT_IMPERATIVE_CUES = (
    "walk me through", "talk to me about", "tell me about", "describe",
    "explain", "elaborate", "go ahead", "let's discuss", "i'm curious",
    "i am curious", "i'd like to hear", "i would like to hear", "show me",
    "take me through", "give me an example", "share your", "your thoughts on",
)
# Trailing-cue: an interviewer prompt that hangs expecting completion.
LIVE_IMPLICIT_TRAILING_CUES = ("so...", "and...", "because...", "which means", "so basically")

# Hypothetical / assumption scenario probes ("Suppose the DB goes down.",
# "Let's say we have a million users.") — these expect the candidate to
# respond even with no wh-word and no '?'. Single-word cues are
# pronoun-guarded in implicit.detect_hypothetical ("I suppose…" is a hedge,
# not a probe); phrase cues are unambiguous wherever they appear.
LIVE_HYPOTHETICAL_SINGLE_CUES = (
    "suppose", "assume", "assuming", "imagine", "hypothetically",
)
LIVE_HYPOTHETICAL_PHRASE_CUES = (
    "let's say", "lets say", "what if", "say we have", "say you have",
    "say your", "consider a scenario", "consider the case", "picture this",
    "what would happen if", "how would you handle it if",
    "in a scenario where",
)

# --- live/interrupt.py — interruption / self-correction cues -------------
LIVE_INTERRUPT_CUES = (
    "actually", "wait", "hold on", "hang on", "before that", "scratch that",
    "never mind", "nevermind", "forget that", "let me rephrase", "let's skip",
    "lets skip", "skip that", "instead of", "rather than", "on second thought",
    "let's leave", "lets leave", "leave that",
)
LIVE_INTERRUPT_CORRECTION_CUES = ("i meant", "sorry", "i mean", "correction", "let me correct",
                    "that's wrong", "not what i", "instead")

# --- live/negotiate.py — HR intent cues + no-manipulation guard ----------
# Keys mirror the intent identifiers in app/live/negotiate.py (SALARY, ...).
LIVE_NEGOTIATE_INTENT_CUES = {
    # More-specific intents first (classify returns the FIRST match).
    "low_offer": ("we can offer", "we're offering", "we are offering", "our offer is",
                  "the offer is", "offering you", "we'd offer", "we can do",
                  # Budget pushback: the interviewer says they can't meet the
                  # candidate's expected number — same play (acknowledge, prove
                  # value, counter politely toward the market range).
                  "don't have enough budget", "dont have enough budget",
                  "not enough budget", "enough budget", "budget to pay",
                  "can't pay you", "cant pay you", "can't afford", "cant afford",
                  "afford to pay", "pay you the amount", "pay you that",
                  "amount you are expecting", "amount you're expecting",
                  "can't match", "cant match", "cannot match your",
                  "match your expectation", "meet your expectation",
                  "below your expectation", "less than you expect",
                  "come down on", "lower your expectation", "reduce your expectation",
                  "tight on budget", "limited budget", "out of our range"),
    "value_justification": ("why do you deserve", "why should we pay", "justify your",
                            "what makes you worth", "why are you worth", "prove you're worth"),
    "final_offer": ("final offer", "best we can do", "take it or leave", "this is our final",
                    "cannot go higher", "can't go higher", "highest we can", "non-negotiable"),
    "salary": ("salary", "compensation", "ctc", "package", "expected pay", "how much",
             "expectation"),
    "notice_period": ("notice period", "when can you join", "how soon", "joining date"),
    "counter_offer": ("counter offer", "counter-offer", "another offer", "competing offer"),
    "why_join": ("why do you want to join", "why us", "why this company", "why this role"),
    "why_leaving": ("why are you leaving", "why do you want to leave", "leaving your current"),
    "benefits": ("benefits", "perks", "stock", "equity", "esop", "bonus", "relocation"),
}
# Coercive / deceptive phrasings the guard must never emit.
LIVE_NEGOTIATE_MANIPULATION = (
    "lie", "lying", "fabricate", "make up", "pretend you have", "bluff",
    "fake offer", "exaggerate", "inflate", "threaten", "ultimatum", "deceive",
    "mislead",
)

# --- live/objective.py — evaluation-objective / expected-depth cues ------
LIVE_OBJECTIVE_TRADEOFF_CUES = ("trade-off", "tradeoff", "pros and cons", "vs", "versus",
                  "difference between", "cap theorem", "when would you", "choose between")
LIVE_OBJECTIVE_DESIGN_CUES = ("design", "architecture", "scale", "throughput", "high availability")
LIVE_OBJECTIVE_INTERNALS_CUES = ("internally", "under the hood", "how does it work", "implementation",
                   "internals", "mechanism", "algorithm behind")
LIVE_OBJECTIVE_SOURCE_CUES = ("source code", "line by line", "exact implementation")
LIVE_OBJECTIVE_BEHAVIORAL_CUES = ("tell me about a time", "describe a situation", "conflict")
# Multi-pass (R50) depth-escalation cues.
LIVE_OBJECTIVE_ESCALATE_CUES = ("but why", "go deeper", "in more detail", "more detail",
                  "elaborate", "under the hood", "specifically", "for example")

# --- live/phase.py — interview-phase cue lexicons ------------------------
LIVE_PHASE_INTRO_CUES = ("tell me about yourself", "introduce yourself", "walk me through your",
               "walk me through your resume", "about your background")
LIVE_PHASE_RESUME_CUES = ("on your resume", "in your resume", "your last project",
                "your current project", "your experience with", "your role at")
LIVE_PHASE_DESIGN_CUES = ("design a", "design an", "system design", "architecture", "scale to",
                "how would you scale", "high availability", "throughput", "distributed")
LIVE_PHASE_CODING_CUES = ("implement", "write a function", "write code", "leetcode", "algorithm",
                "time complexity", "data structure", "reverse a", "sort the")
LIVE_PHASE_BEHAVIORAL_CUES = ("tell me about a time", "describe a situation", "how do you handle",
                    "conflict", "disagreement", "a challenge you", "your weakness",
                    "your strength", "why should we hire")
LIVE_PHASE_HR_CUES = ("salary", "compensation", "ctc", "package", "expectations", "notice period",
            "counter offer", "why do you want to join", "why are you leaving",
            # Compensation / budget phrasings (the interviewer pushing back on
            # pay). Kept as reasonably specific substrings to avoid firing on
            # unrelated technical talk ("payload", "time budget", ...).
            "expecting", "expected pay", "expected salary", "pay you",
            "budget to pay", "afford to pay", "afford you", "enough budget",
            "remuneration", "stipend", "take home", "in hand", "pay range",
            "salary range", "your number", "hike", "negotiate", "how much are you",
            "amount you are expecting", "amount you're expecting")
LIVE_PHASE_CLOSING_CUES = ("any questions for us", "do you have any questions", "that's all",
                 "we're done", "wrap up", "thanks for your time")

# --- live/satisfaction.py — interviewer-satisfaction cues ----------------
LIVE_SATISFACTION_CLOSED_CUES = (
    "good", "great", "perfect", "makes sense", "correct", "exactly", "right",
    "got it", "nice", "well done", "that works", "sounds good", "okay good",
    "thank you", "thanks", "cool", "awesome", "fair enough",
)
LIVE_SATISFACTION_OPEN_CUES = (
    "not quite", "not exactly", "think deeper", "are you sure", "really?",
    "hmm", "try again", "what else", "go deeper", "is that all", "not right",
    "incorrect", "that's not", "thats not", "elaborate", "more detail",
)

# --- live/strategy.py — answer-strategy selection cues --------------------
LIVE_STRATEGY_COMPARE_CUES = ("difference between", " vs ", " versus ", "compare", "compared to",
                 "better than", "or ")
LIVE_STRATEGY_TRADEOFF_CUES = ("trade-off", "tradeoff", "trade off", "pros and cons", "pros & cons",
                  "advantages and disadvantages", "when would you use")
LIVE_STRATEGY_DEBUG_CUES = ("debug", "not working", "throwing an error", "fails", "why is this",
               "what's wrong", "fix this")
# Definition-strategy concept prefixes (question head words).
LIVE_STRATEGY_CONCEPT_PREFIXES = ("what is", "what are", "explain", "define")

# --- live/rhetorical.py — rhetorical-question disambiguation --------------
# High-confidence rhetorical tags (the whole utterance, roughly).
LIVE_RHETORICAL_TAGS = (
    "right?", "right ?", "make sense?", "makes sense?", "you know?",
    "you know what i mean?", "okay?", "ok?", "got it?", "isn't it?",
    "doesn't it?", "agreed?", "see?", "capisce?", "yeah?",
)
# Self-answered lead-in: a question immediately followed by its own answer.
LIVE_RHETORICAL_SELF_ANSWER_CONNECTORS = (" well, ", " so, ", " basically, ", " the answer is ",
                           " it's ", " it is ")

# --- live/premise.py — false-premise detection ----------------------------
# Absolute claims are over-strong and often the planted false premise.
LIVE_PREMISE_ABSOLUTES = ("only", "always", "never", "cannot", "can't", "must", "guarantees",
              "impossible", "every", "no way", "all of", "none of")
# Confirmation tags inviting a yes.
LIVE_PREMISE_CONFIRM_TAGS = ("right?", "correct?", "isn't it?", "aren't they?", "yes?",
                 "true?", "don't you think", "wouldn't you say", "no?")

# --- live/contradiction.py — challenge / temporal-reference cues ----------
LIVE_CONTRADICTION_CHALLENGE_CUES = (
    "but you said", "you just said", "earlier you said", "didn't you say",
    "isn't that contradictory", "that contradicts", "but earlier", "you claimed",
    "a moment ago you", "but you mentioned", "that's not what you said",
)
LIVE_CONTRADICTION_TEMPORAL_CUES = (
    "earlier", "before", "a moment ago", "previously", "back when",
    "you said earlier", "go back to", "back to", "minutes ago", "at the start",
)

# --- live/world_model.py — assumption / constraint extraction cues --------
LIVE_WORLD_ASSUMPTION_CUES = ("assume", "let's say", "lets say", "suppose", "imagine",
                    "say we have", "given that")
LIVE_WORLD_CONSTRAINT_CUES = ("must", "should not", "shouldn't", "cannot", "can't use",
                    "without using", "no third-party", "constraint is",
                    "you can only", "limited to", "budget is")

# --- live/revise.py — answer-revision (reinterpretation) cues -------------
LIVE_REVISE_REINTERPRET_CUES = (
    "i meant", "i mean", "i was asking about", "i was referring to",
    "what i meant", "to clarify i meant", "sorry i meant", "no i meant",
    "not that", "not what i", "i was talking about",
)
