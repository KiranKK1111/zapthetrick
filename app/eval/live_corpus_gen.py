"""Generate the large Live question-detection corpus.

The seed corpus was 32 rows. At that size a single row moves F1 by ~3 points, so
no measured difference between two builds could be trusted — you could not tell
an improvement from noise. This generator builds a few thousand annotated
interviewer utterances from a deliberate taxonomy so the metrics mean something.

Design rules, because a bad corpus is worse than a small one:

* **Deterministic.** No RNG. The same tree of templates × topics every run, so a
  metric change is always a code change, never a sampling accident.
* **Adversarial by construction.** The negatives are not padding. They are the
  cases that actually cause a false answer in a real interview: imperatives that
  are logistics rather than prompts ("give me one moment"), agenda-setting that
  opens with an interrogative word, thinking-aloud, and interviewer self-talk.
  A corpus of easy negatives would report a flattering false-answer rate and
  teach nothing.
* **ASR-realistic.** Roughly half the questions appear WITHOUT a '?', because
  speech-to-text drops terminal punctuation constantly and that is exactly where
  detection gets hard.
* **Labelled by construction.** Each row carries the template family that
  produced it, so a failure report says *which kind* of utterance broke rather
  than just listing strings.
"""
from __future__ import annotations

import json
import pathlib

# ── Topic vocabulary ────────────────────────────────────────────────────────
# Real interview subject matter, so the lexical surface is representative.

TOPICS = [
    "Kafka", "a hashmap", "polymorphism", "the JVM garbage collector",
    "database indexing", "TCP handshakes", "REST versus gRPC", "Docker",
    "Kubernetes", "microservices", "event sourcing", "CAP theorem",
    "eventual consistency", "database sharding", "connection pooling",
    "the actor model", "dependency injection", "unit testing",
    "continuous integration", "blue-green deployment", "OAuth",
    "JWT expiry", "SQL injection", "cross-site scripting", "TLS termination",
    "load balancing", "CDN caching", "Redis persistence", "B-tree indexes",
    "write-ahead logging", "two-phase commit", "the saga pattern",
    "idempotency keys", "backpressure", "circuit breakers", "rate limiting",
    "the observer pattern", "immutability", "memory leaks", "race conditions",
    "deadlock detection", "thread pools", "async I/O", "the event loop",
    "virtual memory", "cache invalidation", "consistent hashing",
    "bloom filters", "quicksort", "binary search trees", "graph traversal",
    "dynamic programming", "big-O analysis", "recursion", "linked lists",
    "normalization", "foreign keys", "transactions isolation levels",
    "message queues", "webhooks", "GraphQL resolvers", "server-side rendering",
    "state management", "code review", "technical debt", "pair programming",
]

SKILLS = [
    "Python", "Java", "Go", "TypeScript", "Rust", "SQL", "React", "Spring Boot",
    "PostgreSQL", "MongoDB", "Terraform", "AWS Lambda", "CI pipelines",
]

# ── Question templates ──────────────────────────────────────────────────────
# `{}` is the topic. Family names appear in the corpus so failures are grouped.

WH_QUESTIONS = [
    "What is {}",
    "What are the tradeoffs of {}",
    "How does {} work",
    "How would you explain {} to a junior developer",
    "Why would you use {}",
    "Why does {} matter in production",
    "When would you reach for {}",
    "Where does {} break down at scale",
    "Which part of {} is most often misunderstood",
    "What problem does {} actually solve",
    "How do you debug an issue with {}",
    "What happens under the hood in {}",
]

YESNO_QUESTIONS = [
    "Is {} something you have worked with",
    "Are there downsides to {}",
    "Does {} scale horizontally",
    "Do you think {} is overused",
    "Can {} handle a sudden traffic spike",
    "Could you walk me through {}",
    "Would you choose {} for a greenfield project",
    "Have you had to tune {} before",
    "Should every team adopt {}",
]

IMPERATIVE_PROMPTS = [
    "Tell me about {}",
    "Explain {}",
    "Describe how you would implement {}",
    "Walk me through {}",
    "Give me an example of {}",
    "Talk me through your understanding of {}",
    "Take me through the internals of {}",
    "Compare {} with the alternatives",
]

INDIRECT_QUESTIONS = [
    "I would like to hear your thoughts on {}",
    "I am curious how you would approach {}",
    "Maybe you could say a little about {}",
    "It would be great if you could cover {}",
    "Let us talk about {}",
    "I want to understand your experience with {}",
]

SCENARIO_QUESTIONS = [
    "Suppose traffic doubles overnight, how would {} hold up",
    "Imagine the primary database fails, what happens to {}",
    "Say you inherit a system using {}, where do you start",
    "If latency spiked tomorrow, how would you tell whether {} was the cause",
    "Let us say the team disagrees about {}, how do you resolve it",
]

COMPARISON_QUESTIONS = [
    "What is the difference between {} and the usual alternative",
    "How does {} compare to what you used at your last job",
    "{} versus doing it by hand, which wins",
]

BEHAVIORAL_QUESTIONS = [
    "Tell me about a time you had to debug something involving {}",
    "Describe a situation where {} caused you real trouble",
    "Give me an example of a disagreement you had about {}",
    "What is the hardest problem you have solved with {}",
]

CODING_QUESTIONS = [
    "Write a function that validates {}",
    "Implement {} from scratch",
    "Given an array of integers, return the two that sum to a target",
    "How would you code a rate limiter for {}",
    "Can you sketch the data model for {}",
]

FOLLOWUP_FRAGMENTS = [
    "And what about {}",
    "Why not {}",
    "What if we removed {}",
    "How so",
    "Such as",
    "Can you go deeper on {}",
    "And in production",
]

# ── Non-question templates — the adversarial half ───────────────────────────

LOGISTICS = [
    "Give me one moment, my screen froze",
    "Give me a second, I need to let someone in",
    "Let me share my screen",
    "Let me pull up your resume",
    "Hold on, my camera is acting up",
    "One second, I am losing you a little",
    "Bear with me, the room booking ran over",
    "Sorry, I had you on mute",
    "Let me just grab the coding link",
    "Give me a minute to find the right document",
    "I am going to drop the exercise in the chat",
    "Let me get my colleague on the call",
    "Apologies, my connection is unstable",
    "Just a moment while I start the recording",
    "Let me turn my video off to save bandwidth",
]

AGENDA = [
    "Before we dive in, I want to describe the interview format",
    "So today we will spend about forty five minutes together",
    "First I will introduce myself, then we will get into the technical part",
    "We have three sections planned for this conversation",
    "I will leave ten minutes at the end for your questions",
    "Let me set some context before we begin",
    "The way this usually works is I ask, you think out loud",
    "We will start easy and ramp up from there",
    "I want to cover architecture first and then coding",
    "Just so you know, there are no trick questions here",
    "After this there will be one more round with the team",
    "Feel free to interrupt me at any point",
]

EXPLANATION = [
    "In our organization we use {} extensively",
    "We mainly build our services around {}",
    "Our team migrated away from {} last year",
    "Most of our stack is {} these days",
    "The platform team owns {} internally",
    "We had a lot of incidents caused by {}",
    "Historically we avoided {} for good reasons",
    "That role would sit close to {}",
]

ACKNOWLEDGEMENT = [
    "Okay, that makes sense",
    "Right, good",
    "Yeah exactly",
    "Mm hmm",
    "Got it, thank you",
    "Perfect, that is what I was looking for",
    "Sure, understood",
    "Nice, that is a good answer",
    "Interesting",
    "Fair enough",
    "That tracks",
    "Great, thanks for walking through that",
]

TRANSITION = [
    "Okay, let us move on",
    "Alright, switching gears",
    "Good, that covers that area",
    "Let us park that and come back to it",
    "Moving to the next topic",
    "We can leave it there",
    "I think we have enough on that one",
]

SELF_TALK = [
    "I am just making a quick note",
    "I am writing that down",
    "Let me think about how to phrase this",
    "I always ask this one and it never gets old",
    "My colleague usually runs this section",
    "I used to work on something similar myself",
    "That reminds me of a project I did",
]

SMALLTALK = [
    "Good morning, thanks for making the time",
    "Hope the weather is better where you are",
    "Nice to finally put a face to the name",
    "Thanks for being flexible with the reschedule",
    "I hope you found the office alright",
    "It has been a busy week here",
]

# ── Multi-question utterances ───────────────────────────────────────────────

MULTI = [
    ("What is {}, why is it used, and how would you scale it",
     ["What is {}", "why is it used", "how would you scale it"]),
    ("Have you used {} before, and if so what went wrong",
     ["Have you used {} before", "what went wrong"]),
    ("How does {} work and when would you avoid it",
     ["How does {} work", "when would you avoid it"]),
    ("Can you define {} and give me a concrete example",
     ["Can you define {}", "give me a concrete example"]),
]


def _rows() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()

    def add(text: str, is_q: bool, family: str, questions=None, topic=""):
        t = text.strip()
        if not t or t in seen:
            return
        seen.add(t)
        out.append({
            "text": t,
            "is_question": is_q,
            "questions": questions if questions is not None else ([t] if is_q else []),
            "topic": topic,
            "source": family,
            "note": family,
        })

    positive_families = [
        ("wh", WH_QUESTIONS),
        ("yesno", YESNO_QUESTIONS),
        ("imperative", IMPERATIVE_PROMPTS),
        ("indirect", INDIRECT_QUESTIONS),
        ("scenario", SCENARIO_QUESTIONS),
        ("comparison", COMPARISON_QUESTIONS),
        ("behavioral", BEHAVIORAL_QUESTIONS),
        ("coding", CODING_QUESTIONS),
    ]

    # Questions × topics. Alternate '?' on and off: STT drops terminal
    # punctuation constantly, and an un-'?'-ed question is the hard case.
    for family, templates in positive_families:
        for ti, template in enumerate(templates):
            for pi, topic in enumerate(TOPICS):
                text = template.format(topic)
                punct = "?" if (ti + pi) % 2 == 0 else ""
                add(text + punct, True, family, topic=topic.strip("the "))

    # Follow-up fragments — short, context-dependent, easy to miss.
    for ti, template in enumerate(FOLLOWUP_FRAGMENTS):
        for pi, topic in enumerate(TOPICS[:20]):
            text = template.format(topic) if "{}" in template else template
            add(text + ("?" if (ti + pi) % 2 == 0 else ""), True, "followup",
                topic=topic.strip("the "))

    # Skill-specific direct questions.
    for si, skill in enumerate(SKILLS):
        for template in ("How much {} have you written",
                         "Rate your comfort with {}",
                         "What do you like least about {}",
                         "Would you be happy working in {} daily"):
            add(template.format(skill) + ("?" if si % 2 == 0 else ""),
                True, "skill", topic=skill)

    # Multi-question utterances.
    for template, parts in MULTI:
        for topic in TOPICS[:30]:
            add(template.format(topic) + "?", True, "multi",
                questions=[p.format(topic) for p in parts], topic=topic)

    # ── Negatives ──
    for family, bank in (("logistics", LOGISTICS), ("agenda", AGENDA),
                         ("acknowledgement", ACKNOWLEDGEMENT),
                         ("transition", TRANSITION), ("self_talk", SELF_TALK),
                         ("smalltalk", SMALLTALK)):
        for text in bank:
            add(text, False, family)

    for template in EXPLANATION:
        for topic in TOPICS:
            text = template.format(topic) if "{}" in template else template
            add(text, False, "explanation", topic=topic.strip("the "))

    return out


def write(path: str | pathlib.Path) -> int:
    rows = _rows()
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write("# Live question-detection corpus — GENERATED by "
                 "app/eval/live_corpus_gen.py. Do not hand-edit; change the "
                 "generator so the taxonomy stays explicit and reproducible.\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    target = pathlib.Path(__file__).parent / "data" / "live_corpus_large.jsonl"
    n = write(target)
    print(f"wrote {n} rows to {target}")


__all__ = ["write", "TOPICS", "SKILLS"]
