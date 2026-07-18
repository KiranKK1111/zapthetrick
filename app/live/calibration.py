"""
Seniority-band calibration for live interview answers
(live-conversational-intelligence — profile-aware answer framing).

Classifies the candidate into a seniority BAND and a career TRACK from the
resume profile (years of experience, current / target title, project & skill
signals) using the industry ladder documented in `BandSpecific.md`, then emits
an ANSWER GUIDANCE directive so the spoken answer is pitched at the right level
— from a sharp fresher to a highly professional senior/staff voice.

"Intelligent both" calibration (the two things the answer must balance):
  * real_band   — the candidate's ACTUAL band, inferred from their resume. This
                  is the truthful floor: NEVER fabricate seniority, titles,
                  employers, or years beyond what the resume supports.
  * target_band — the band of the ROLE they are interviewing for.
  The directive pitches the answer at the real band, but FRAMES it toward the
  target band's expected capabilities wherever the candidate has genuine,
  relevant strengths — bridging the gap with transferable experience and a clear
  growth trajectory, without claiming anything untrue. When the candidate already
  meets or exceeds the target, it tells the model to show that depth outright.

Deterministic + fail-open: pure functions, no I/O, never raises. On any error
`build_calibration` returns None and `calibration_directive` returns "" so the
answer path is completely unaffected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Seniority ladder — a compact, faithful projection of the B0–B13 / L0–L10
# ladder in BandSpecific.md onto 8 framing tiers. `low_years` is the typical
# lower bound of professional experience for the band; `focus` / `ownership` /
# `emphasis` shape how the answer should sound at that tier.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Band:
    index: int
    slug: str
    label: str
    low_years: float          # typical lower bound of experience (years)
    focus: str                # the band's primary scope of work
    ownership: str            # ownership language appropriate to the band
    emphasis: str             # capability dimensions to foreground
    caution: str              # what NOT to overclaim at this band


SENIORITY_BANDS: tuple[Band, ...] = (
    Band(0, "intern", "Intern / Trainee", 0.0,
         "learning fundamentals and shipping small, well-scoped tasks with guidance",
         "contributed to / helped build, under mentorship",
         "eagerness to learn, fundamentals (data structures, clean code), curiosity, coursework and hands-on projects",
         "leadership, architecture ownership, or team/mentoring claims"),
    Band(1, "fresher", "Fresher / Graduate Engineer", 0.0,
         "building features with guidance and growing fast",
         "built / implemented, with support from seniors",
         "solid fundamentals, quick learning, personal & academic projects, enthusiasm, coachability",
         "leading teams, setting architecture, or years of production ownership"),
    Band(2, "junior", "Junior / Associate Engineer", 1.0,
         "owning small modules end to end and handling routine ambiguity",
         "owned / delivered, with occasional guidance",
         "reliable execution, testing, debugging, collaboration, incremental design decisions",
         "org-wide impact, cross-team leadership, or deep architecture ownership"),
    Band(3, "mid", "Software Engineer (Mid-level)", 3.0,
         "owning features and modules independently and making local design choices",
         "designed and owned / drove",
         "independent execution, sound trade-offs within a module, code quality, mentoring interns lightly, delivery under deadlines",
         "company-wide architecture or people-management scope"),
    Band(4, "senior", "Senior Software Engineer", 5.0,
         "designing features across modules, mentoring, and owning quality and delivery",
         "led the design of / drove / owned end to end",
         "system design trade-offs, mentoring, cross-module ownership, reliability, measurable business impact, technical judgement",
         "org-wide or multi-team strategy beyond what was actually held"),
    Band(5, "lead", "Lead / Staff Engineer", 7.0,
         "leading projects and teams, and owning cross-team architecture",
         "led / architected / drove across teams",
         "architecture and scalability, cross-team leadership, mentoring engineers, aligning technical work to business outcomes, ambiguity at scale",
         "executive / company-wide mandate not actually held"),
    Band(6, "principal", "Principal / Architect", 10.0,
         "setting organization-wide technical direction and architecture",
         "set direction for / defined the architecture of",
         "org-wide technical strategy, systems architecture, deep trade-off reasoning, influence without authority, long-horizon impact",
         "board / C-suite business ownership unless it genuinely applies"),
    Band(7, "distinguished", "Distinguished / Engineering Leadership", 15.0,
         "driving company-wide technical and business strategy",
         "defined the strategy for / drove company-wide",
         "company-wide influence, business + technical strategy, building and scaling teams and platforms, industry-level perspective",
         "nothing — but keep claims concrete and grounded in real outcomes"),
)

_BY_SLUG = {b.slug: b for b in SENIORITY_BANDS}
_MAX_INDEX = SENIORITY_BANDS[-1].index

# Title keywords → the MINIMUM band the title implies. A title is a strong,
# resume-backed signal, so it can lift the years-based estimate (never used to
# fabricate downward below a clearly stated experience).
_TITLE_BANDS: tuple[tuple[str, int], ...] = (
    ("intern", 0), ("trainee", 0), ("graduate engineer", 1), ("graduate trainee", 0),
    ("fresher", 1), ("associate", 2), ("junior", 2), ("jr.", 2), ("jr ", 2),
    ("sde-1", 2), ("sde 1", 2), ("sde i", 2), ("sde1", 2),
    ("engineer i", 2), ("engineer 1", 2), ("developer i", 2),
    ("sde-2", 3), ("sde 2", 3), ("sde ii", 3), ("sde2", 3),
    ("engineer ii", 3), ("engineer 2", 3), ("developer ii", 3),
    ("mid-level", 3), ("mid level", 3),
    # "Member of Technical Staff" ladder (Salesforce / VMware / AMD style).
    ("member of technical staff", 3), ("mts", 3), ("smts", 4), ("lmts", 5), ("pmts", 6),
    ("senior", 4), ("sr.", 4), ("sr ", 4), ("sde-3", 4), ("sde 3", 4), ("sde iii", 4),
    ("engineer iii", 4), ("developer iii", 4), ("founding engineer", 4),
    ("lead", 5), ("staff", 5), ("tech lead", 5), ("technical lead", 5),
    ("team lead", 5), ("engineering lead", 5),
    ("principal", 6), ("architect", 6), ("senior staff", 6),
    ("distinguished", 7), ("fellow", 7), ("director", 7), ("head of", 7),
    ("vp ", 7), ("vice president", 7), ("cto", 7), ("chief", 7),
    # People-management ladder maps onto the leadership tiers.
    ("engineering manager", 5), ("senior engineering manager", 6),
    ("senior manager", 6), ("group manager", 6), ("manager", 4),
)

# --------------------------------------------------------------------------- #
# Career tracks — detected from the target role + skills. Each track adds a line
# of vocabulary / dimension emphasis so the framing is domain-appropriate, not
# just generic "software engineer".
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Track:
    slug: str
    label: str
    emphasis: str


TRACKS: tuple[Track, ...] = (
    Track("ai_ml", "AI / ML Engineering",
          "concrete model/LLM choices and WHY, evaluation methodology and metrics, "
          "RAG and agent orchestration, data quality and drift, and hard latency/cost/"
          "accuracy trade-offs — show engineering judgement, not just API calls"),
    Track("data", "Data Engineering / Science",
          "pipeline and schema design, data correctness/lineage/idempotency, "
          "throughput at scale, warehouse/SQL performance tuning, and the analytics "
          "or product decision the data actually unlocked"),
    Track("cloud", "Cloud / Platform Engineering",
          "the specific cloud primitives used, infrastructure-as-code, reliability and "
          "autoscaling, blast-radius and cost control, and the ergonomics your platform "
          "gave other engineers"),
    Track("devops", "DevOps / SRE",
          "CI/CD pipeline design, observability (metrics/logs/traces), on-call and "
          "incident response with real MTTR/SLO numbers, and automation that removed toil"),
    Track("security", "Security Engineering",
          "threat modelling, secure-by-design decisions, concrete vulnerabilities found "
          "and remediated, defence-in-depth, and balancing risk against developer velocity"),
    Track("frontend", "Frontend / Mobile",
          "component/state architecture, measurable performance (load, jank, bundle size), "
          "accessibility, cross-platform trade-offs, and the UX quality the work delivered"),
    Track("backend", "Backend Engineering",
          "API and service boundaries, data consistency and concurrency, scaling and "
          "failure modes, and the reliability/latency numbers you moved"),
    Track("fullstack", "Full-Stack Engineering",
          "genuine end-to-end ownership from UI through API to data, pragmatic trade-offs "
          "under time pressure, and shipping complete features that moved a product metric"),
    Track("qa", "Quality Engineering / SDET",
          "test strategy and the automation frameworks you built, coverage and flake "
          "reduction with numbers, quality gates in CI, and defects prevented before release"),
    Track("architecture", "Architecture",
          "system and solution design, the cross-cutting trade-offs you weighed, "
          "non-functional requirements (scale, cost, security, maintainability), and "
          "long-horizon decisions and their rationale"),
    Track("product", "Product / Program",
          "the user and business outcome, how you prioritised and made the call under "
          "uncertainty, stakeholder alignment, and the metric your delivery moved"),
    Track("data_science", "Data Science",
          "sharp problem framing, experiment design and statistical rigour, honest model "
          "evaluation and limitations, and how you translated findings into a decision or impact"),
    Track("research", "Research (Research Engineer / Scientist)",
          "the research question and hypothesis, experimental rigour and reproducibility, "
          "novel vs incremental contribution, and — for a research ENGINEER — turning "
          "research into robust, scalable systems rather than one-off notebooks"),
    Track("design", "Design / UX",
          "the user problem and research behind the design, interaction and visual craft, "
          "design-system thinking, accessibility, and how you measured whether the design "
          "actually improved the experience"),
    Track("consulting", "Consulting / Solutions",
          "client problem framing, stakeholder management, pragmatic recommendations under "
          "constraints, and delivering measurable business outcomes — not just technology for "
          "its own sake"),
    Track("sales_eng", "Sales / Solutions Engineering",
          "translating customer needs into solution architecture, credible technical "
          "demos and POCs, objection handling, and partnering with sales to unblock "
          "deals without over-promising"),
    Track("devrel", "Developer Relations / Advocacy",
          "developer empathy and clear technical communication, content and demos that teach, "
          "community and feedback loops back into the product, and adoption/engagement you "
          "actually moved"),
    Track("networking", "Networking / Infrastructure",
          "network and systems fundamentals (routing, DNS, load balancing), automation of "
          "infra, reliability and capacity, and the throughput/latency/uptime numbers you moved"),
    Track("embedded", "Embedded / Firmware",
          "resource and real-time constraints, hardware/software interface and drivers, "
          "power and memory budgets, safety/reliability, and debugging on real devices"),
    Track("gaming", "Graphics / Gaming",
          "the rendering or engine problem, performance at frame budget (GPU/CPU), math and "
          "physics correctness, memory, and the player-facing quality the work delivered"),
    Track("blockchain", "Blockchain / Web3",
          "security and correctness of smart contracts, gas/cost trade-offs, consensus and "
          "trust assumptions, auditability, and the concrete decentralised problem being solved"),
    Track("enterprise", "Enterprise Applications (ERP / CRM)",
          "platform and customisation depth (SAP/Salesforce/ServiceNow/Workday), business-"
          "process mapping, integration with existing systems, and delivering the operational "
          "outcome the business needed"),
)

_TRACK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai_ml": ("ai engineer", "ml engineer", "machine learning", "llm", "genai",
              "generative ai", "agentic", "rag", "nlp", "deep learning", "mlops",
              "prompt", "computer vision", "ai platform", "applied ai"),
    "data": ("data engineer", "analytics engineer", "big data", "etl", "spark",
             "warehouse", "data platform", "streaming data"),
    "cloud": ("cloud engineer", "cloud architect", "platform engineer", "aws",
              "azure", "gcp", "kubernetes", "infrastructure engineer"),
    "devops": ("devops", "sre", "site reliability", "release engineer", "ci/cd",
               "observability"),
    "security": ("security engineer", "appsec", "application security", "penetration",
                 "cyber", "infosec", "devsecops", "red team", "soc analyst"),
    "frontend": ("frontend", "front-end", "front end", "react", "angular", "vue",
                 "flutter", "mobile engineer", "android", "ios", "ui engineer"),
    "backend": ("backend", "back-end", "back end", "api engineer", "server",
                "microservices", "spring", "node", "golang"),
    "fullstack": ("full stack", "full-stack", "fullstack"),
    "qa": ("qa engineer", "sdet", "test engineer", "automation engineer",
           "quality engineer", "performance test"),
    "architecture": ("solution architect", "enterprise architect", "technical architect",
                     "system architect", "ai architect"),
    "product": ("product manager", "program manager", "technical program",
                "delivery manager", "scrum master"),
    "data_science": ("data scientist", "statistician", "bi engineer", "bi developer",
                     "quantitative analyst"),
    "research": ("research engineer", "research scientist", "applied scientist",
                 "applied research", "foundation model", "model training engineer",
                 "distributed training", "ml research", "research lab", "phd researcher",
                 "postdoctoral", "research software engineer"),
    "design": ("ux designer", "ui designer", "product designer", "interaction designer",
               "ux engineer", "ux researcher", "design systems", "motion designer",
               "visual designer"),
    "consulting": ("technology consultant", "cloud consultant", "ai consultant",
                   "solution consultant", "management consultant", "sap consultant",
                   "salesforce consultant", "principal consultant", "managing consultant"),
    "sales_eng": ("sales engineer", "solutions engineer", "solution engineer",
                  "sales engineering", "pre-sales", "presales", "field application engineer"),
    "devrel": ("developer advocate", "developer relations", "devrel",
               "developer evangelist", "technical evangelist", "community engineer",
               "developer experience engineer"),
    "networking": ("network engineer", "network automation", "sd-wan",
                   "networking engineer", "noc engineer", "network architect",
                   "network security engineer"),
    "embedded": ("embedded", "firmware", "device driver", "rtos", "microcontroller",
                 "iot engineer", "automotive software", "autosar", "adas",
                 "avionics", "flight software", "bare-metal"),
    "gaming": ("game engine", "gameplay engineer", "graphics engineer", "gpu engineer",
               "rendering engineer", "game developer", "physics engine", "xr engineer",
               "ar/vr", "vr engineer", "unreal", "unity", "shader"),
    "blockchain": ("blockchain", "smart contract", "web3", "solidity",
                   "cryptography engineer", "defi", "distributed ledger"),
    "enterprise": ("sap", "abap", "salesforce developer", "salesforce architect",
                   "salesforce", "servicenow", "workday", "dynamics 365", "peoplesoft",
                   "mulesoft", "oracle erp", "oracle ebs", "erp developer", "crm developer",
                   "power platform", "outsystems", "mendix", "appian"),
}


# --------------------------------------------------------------------------- #
# People-management ladder (BandSpecific.md lines 50-73 / 283-301 / 610-625).
# The IC `_TITLE_BANDS` already fold management titles onto seniority tiers for
# the numeric real-vs-target math; this ADDS a management-SPECIFIC framing layer
# so a manager's answer foregrounds people leadership, org scope, and strategy —
# distinct from an IC's hands-on ownership language. `band_index` is the nearest
# seniority tier (for gap comparison), not a replacement for it.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManagementTier:
    slug: str
    label: str
    band_index: int          # nearest seniority tier for real-vs-target comparison
    scope: str               # the span of people / org the role leads
    emphasis: str            # leadership dimensions to foreground
    caution: str             # what NOT to overclaim at this management tier


MANAGEMENT_LADDER: tuple[ManagementTier, ...] = (
    ManagementTier("em", "Engineering Manager", 5,
                   "a single engineering team",
                   "unblocking and growing engineers, delivery and execution health, 1:1s and "
                   "performance, translating roadmap into shipped work, and hiring",
                   "director/VP-level org scope, headcount, or company strategy not actually held"),
    ManagementTier("senior_em", "Senior Engineering Manager", 6,
                   "multiple teams or a large team (often managers/leads reporting in)",
                   "developing leads/managers, cross-team roadmap and dependencies, org health and "
                   "process, and connecting execution to business goals",
                   "director/VP mandate, budget ownership, or org design beyond what was held"),
    ManagementTier("director", "Director of Engineering", 7,
                   "a multi-team engineering organisation",
                   "org design and headcount, hiring and manager development, cross-org delivery, "
                   "budget, and setting technical + people strategy for the org",
                   "VP/C-suite company-wide ownership unless it genuinely applies"),
    ManagementTier("vp", "VP of Engineering", 7,
                   "a major engineering organisation and its leaders",
                   "executive strategy and org design, budget and headcount at scale, building the "
                   "leadership bench, and aligning engineering to company outcomes",
                   "founder/CEO-level business ownership unless it genuinely applies"),
    ManagementTier("cto", "CTO / VP+ Technology Leadership", 7,
                   "the company's technology function",
                   "company-wide technology vision and strategy, executive stakeholder and board "
                   "communication, org and platform scaling, and business + technical bets",
                   "nothing — but keep every claim concrete and grounded in real outcomes"),
)

_MGMT_BY_SLUG = {m.slug: m for m in MANAGEMENT_LADDER}

# Management title cues → tier slug. Matched with WORD BOUNDARIES so short cues
# like "cto"/"vp" don't fire inside words (e.g. "cto" hiding in "dire-cto-r").
_MANAGEMENT_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("chief technology officer", "cto"), ("cto", "cto"), ("chief architect", "cto"),
    ("chief technical officer", "cto"), ("chief engineering officer", "cto"),
    ("svp", "cto"), ("senior vice president", "cto"), ("executive vice president", "cto"),
    ("vice president of engineering", "vp"), ("vp of engineering", "vp"),
    ("vp engineering", "vp"), ("vice president", "vp"), ("vp", "vp"),
    ("senior director", "director"), ("director of engineering", "director"),
    ("engineering director", "director"), ("director", "director"),
    ("senior engineering manager", "senior_em"), ("senior manager, engineering", "senior_em"),
    ("group engineering manager", "senior_em"), ("group manager", "senior_em"),
    ("engineering manager", "em"), ("software engineering manager", "em"),
    ("dev manager", "em"), ("development manager", "em"),
)


def detect_management(title: str) -> ManagementTier | None:
    """Detect a people-management tier from a job title. None for IC titles.
    Returns the HIGHEST tier any cue implies (so 'Senior Director' beats
    'Director'). Matched on word boundaries so IC titles and unrelated words
    never falsely trigger a management framing."""
    if not title:
        return None
    t = title.lower()
    best: ManagementTier | None = None
    for kw, slug in _MANAGEMENT_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", t):
            tier = _MGMT_BY_SLUG.get(slug)
            if tier is not None and (best is None or tier.band_index > best.band_index):
                best = tier
    return best


# --------------------------------------------------------------------------- #
# Industry vertical — the 4th taxonomy dimension (BandSpecific.md lines 919-928:
# domain × specialization × seniority × INDUSTRY). Seniority (bands) and
# specialization (tracks) already exist; this adds a light industry-context hint
# detected from the role + JD + skills so examples can be grounded in the
# vertical's real constraints and vocabulary — without claiming domain years the
# resume does not show.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Industry:
    slug: str
    label: str
    hint: str                # what a strong answer in this vertical foregrounds


INDUSTRIES: tuple[Industry, ...] = (
    Industry("fintech", "FinTech / Financial Services",
             "correctness and auditability of money movement, regulatory/compliance "
             "constraints (PCI, KYC/AML), latency and idempotency, and risk"),
    Industry("healthtech", "HealthTech / Healthcare",
             "patient-data privacy and compliance (HIPAA/FHIR/interoperability), safety, "
             "reliability, and clinical correctness"),
    Industry("edtech", "EdTech / Education",
             "learner outcomes and engagement, accessibility and scale for many concurrent "
             "users, and content/assessment integrity"),
    Industry("ecommerce", "E-commerce / Retail",
             "catalog, cart and checkout reliability, payments, peak-traffic scaling, "
             "search/recommendation relevance, and conversion"),
    Industry("gaming_ind", "Gaming / Interactive Entertainment",
             "frame-budget performance, player experience and latency, live-ops scale, "
             "and anti-cheat/economy integrity"),
    Industry("telecom", "Telecom / Networking",
             "carrier-grade reliability and scale, protocols and OSS/BSS, latency, and "
             "5G/network-function constraints"),
    Industry("automotive", "Automotive / Mobility",
             "functional safety (ISO 26262/ASIL), real-time and embedded constraints, "
             "reliability, and over-the-air update integrity"),
    Industry("enterprise_ind", "Enterprise SaaS / B2B",
             "multi-tenancy and isolation, integration with customer systems, SLAs and "
             "configurability, and enterprise security/compliance"),
    Industry("cybersec_ind", "Security / Cyber",
             "threat models and adversarial thinking, defence-in-depth, compliance, and "
             "balancing risk against velocity"),
    Industry("govtech", "Government / Public Sector",
             "compliance and accreditation, accessibility, auditability, and long-lived "
             "reliability under strict procurement/security constraints"),
    Industry("media", "Media / Streaming",
             "high-throughput content delivery and CDN, encoding/latency, scale to large "
             "concurrent audiences, and playback quality"),
)

_INDUSTRY_BY_SLUG = {i.slug: i for i in INDUSTRIES}

_INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fintech": ("fintech", "financial services", "banking", "payments", "payment ",
                "trading", "capital markets", "insurtech", "lending", "wealth",
                "neobank", "brokerage", "kyc", "aml", "pci"),
    "healthtech": ("healthtech", "healthcare", "health care", "medtech", "clinical",
                   "hospital", "patient", "hipaa", "fhir", "genomics", "bioinformatics",
                   "pharma", "life sciences", "digital health"),
    "edtech": ("edtech", "education technology", "e-learning", "elearning", "learning platform",
               "lms", "online learning", "education "),
    "ecommerce": ("e-commerce", "ecommerce", "retail", "marketplace", "shopify",
                  "magento", "commerce platform", "checkout", "point of sale"),
    "gaming_ind": ("gaming", "game studio", "video game", "esports", "aaa game",
                   "mobile gaming", "game publisher"),
    "telecom": ("telecom", "telecommunications", "5g", "oss/bss", "carrier",
                "network operator", "mobile network"),
    "automotive": ("automotive", "adas", "autonomous vehicle", "self-driving", "autosar",
                   "ev ", "electric vehicle", "mobility platform", "infotainment"),
    "enterprise_ind": ("enterprise saas", "b2b saas", "enterprise software", "erp",
                       "crm platform", "workday", "servicenow", "salesforce platform"),
    "cybersec_ind": ("cybersecurity", "cyber security", "security vendor", "threat intelligence",
                     "soc ", "siem", "endpoint security"),
    "govtech": ("govtech", "government", "public sector", "defense", "defence",
                "federal", "civic tech"),
    "media": ("media streaming", "streaming platform", "ott", "video streaming", "cdn",
              "broadcast", "digital media"),
}


def detect_industry(role: str, jd: str | None = None,
                    skills: list[str] | None = None) -> Industry | None:
    """Detect the industry vertical from role + JD + skills. JD text is the
    strongest signal (it usually names the domain). None if nothing matches, so
    the directive stays industry-agnostic. First match by declaration order."""
    hay = " ".join(filter(None, [
        (role or "").lower(),
        (jd or "").lower(),
        " ".join(str(s).lower() for s in (skills or [])),
    ]))
    if not hay.strip():
        return None
    for slug, kws in _INDUSTRY_KEYWORDS.items():
        for kw in kws:  # noqa: SIM110
            if kw in hay:
                return _INDUSTRY_BY_SLUG.get(slug)
    return None


@dataclass
class Calibration:
    real_band: Band
    target_band: Band | None = None
    track: Track | None = None
    overridden: bool = False
    management: ManagementTier | None = None
    industry: Industry | None = None
    readiness: str | None = None       # capability-over-title signal (career.py)
    signals: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parsing / classification helpers
# --------------------------------------------------------------------------- #
def _parse_years(value) -> float | None:
    """Extract a years-of-experience number from '5', 5, '5+', '3-4 years', etc."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        m = re.search(r"\d+(?:\.\d+)?", value)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _band_from_years(years: float) -> int:
    """Map years of experience onto a band index using the ladder's low bounds."""
    idx = 1  # default to fresher, not intern, when years are known but tiny
    for b in SENIORITY_BANDS:
        if years >= b.low_years:
            idx = b.index
    return idx


# --------------------------------------------------------------------------- #
# Company-specific NUMERIC internal levels (BandSpecific.md lines ~1519-1534).
# Many companies use private level ladders instead of public titles. Each maps
# onto the 8 framing bands. Bare "L#" is ambiguous across companies, so the
# Amazon ladder is only used when the resume actually names Amazon; otherwise a
# bare "L#" defaults to the FAANG-standard (Google-style) ladder.
# --------------------------------------------------------------------------- #
_LVL_GOOGLE = {3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 7}
_LVL_AMAZON = {4: 2, 5: 3, 6: 4, 7: 6, 8: 6, 9: 7, 10: 7}
_LVL_META = {3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 7}          # E3–E9
_LVL_APPLE = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6}                     # ICT2–ICT6
_LVL_MSFT = {59: 3, 60: 3, 61: 3, 62: 4, 63: 4, 64: 5, 65: 6,   # 59–68+
             66: 6, 67: 6, 68: 7, 69: 7, 70: 7, 80: 7}
_LVL_IBM = {6: 2, 7: 3, 8: 4, 9: 5, 10: 6}                      # Band 6–10
_LVL_IC = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 6}                  # Oracle/NVIDIA/ServiceNow IC#


def _band_from_company_level(title: str) -> int | None:
    """Recognise a company's internal numeric level (Google L#, Meta E#, Amazon
    L#/SDE, Apple ICT#, Microsoft 59-67+, IBM Band #, Oracle/NVIDIA IC#) and map
    it onto a framing band. Returns the highest band any recognised token
    implies, or None. Distinctive prefixes (E#, ICT#, IC#, Band #) are mapped
    directly; the ambiguous bare L# uses Amazon's ladder only when Amazon is
    named, else the Google-standard ladder."""
    if not title:
        return None
    t = f" {title.lower()} "
    best: int | None = None

    def _bump(idx: int | None) -> None:
        nonlocal best
        if idx is not None:
            best = idx if best is None else max(best, idx)

    # Apple ICT# (check before the generic IC# so "ict4" isn't read as "ic").
    for m in re.finditer(r"\bict\s*(\d+)\b", t):
        _bump(_LVL_APPLE.get(int(m.group(1))))
    # Meta E-ladder E3–E9 (distinctive; "e2e" etc. won't match the \b…\b form).
    for m in re.finditer(r"\be(\d+)\b", t):
        _bump(_LVL_META.get(int(m.group(1))))
    # Google / Amazon L-ladder. Amazon only when explicitly named.
    _ltbl = _LVL_AMAZON if "amazon" in t else _LVL_GOOGLE
    for m in re.finditer(r"\bl(\d+)\b", t):
        _bump(_ltbl.get(int(m.group(1))))
    # Oracle / NVIDIA / ServiceNow IC# (skips ICT# handled above).
    for m in re.finditer(r"\bic\s*(\d+)\b", t):
        _bump(_LVL_IC.get(int(m.group(1))))
    # IBM Bands.
    for m in re.finditer(r"\bband\s*(\d+)\b", t):
        _bump(_LVL_IBM.get(int(m.group(1))))
    # Microsoft numeric levels — only when Microsoft is named (bare 2-digit
    # numbers are otherwise far too ambiguous to trust).
    if "microsoft" in t:
        for m in re.finditer(r"\b(\d{2,3})\b", t):
            _bump(_LVL_MSFT.get(int(m.group(1))))
    return best


def _band_from_title(title: str) -> int | None:
    """Highest band implied by any keyword OR company numeric level in a job
    title. None if nothing matches."""
    if not title:
        return None
    t = f" {title.lower()} "
    best: int | None = None
    for kw, idx in _TITLE_BANDS:
        if kw in t:
            best = idx if best is None else max(best, idx)
    lvl = _band_from_company_level(title)
    if lvl is not None:
        best = lvl if best is None else max(best, lvl)
    return best


def _signal_score(cp) -> int:
    """A small, bounded seniority nudge from profile richness (projects / skills /
    achievements). Returns 0, 1, or 2 — never enough to leap multiple bands."""
    try:
        n_proj = len(getattr(cp, "projects", []) or [])
        n_skill = len(getattr(cp, "skills", []) or [])
        n_ach = len(getattr(cp, "achievements", []) or [])
        score = n_proj + n_skill * 0.5 + n_ach
        if score >= 14:
            return 2
        if score >= 7:
            return 1
        return 0
    except Exception:  # noqa: BLE001
        return 0


def _clamp(idx: int) -> int:
    return max(0, min(_MAX_INDEX, idx))


def detect_track(role: str, skills: list[str] | None = None) -> Track | None:
    """Detect the career track from the target role first, then skills. Returns
    None if nothing matches (the directive then stays track-agnostic)."""
    hay = (role or "").lower()
    if skills:
        hay += " " + " ".join(str(s).lower() for s in skills)
    if not hay.strip():
        return None
    # Role text is the strongest signal; check specific tracks before generic
    # backend/fullstack so "AI engineer" doesn't fall through to "engineer".
    order = ("ai_ml", "research", "data_science", "data", "blockchain", "gaming",
             "embedded", "security", "networking", "devops", "cloud", "architecture",
             "enterprise", "consulting", "sales_eng", "devrel", "design", "qa",
             "product", "frontend", "backend", "fullstack")
    by_slug = {t.slug: t for t in TRACKS}
    for slug in order:
        for kw in _TRACK_KEYWORDS.get(slug, ()):  # noqa: SIM110
            if kw in hay:
                return by_slug.get(slug)
    return None


def classify_real_band(profile: dict | None, cp=None,
                       override: str | None = None) -> tuple[Band, dict]:
    """The candidate's ACTUAL band. Manual override wins; otherwise infer from
    years + current title + a small richness nudge. Returns (band, signals)."""
    signals: dict = {}
    # Manual override from the interview setup dialog (session metadata).
    if override:
        ov = override.strip().lower()
        if ov and ov != "auto" and ov in _BY_SLUG:
            signals["override"] = ov
            return _BY_SLUG[ov], signals

    prof = profile if isinstance(profile, dict) else {}
    years = _parse_years(prof.get("years_experience"))
    title = str(prof.get("current_role") or prof.get("headline") or "")

    years_idx = _band_from_years(years) if years is not None else None
    title_idx = _band_from_title(title)
    signals["years"] = years
    signals["years_band"] = years_idx
    signals["title_band"] = title_idx

    if years_idx is None and title_idx is None:
        # No experience signal at all → treat as fresher (safe, truthful floor).
        base = 1
    elif years_idx is None:
        base = title_idx  # type: ignore[assignment]
    elif title_idx is None:
        base = years_idx
    else:
        # Both present: the title is resume-backed, so let it lift the estimate,
        # but don't let a lofty title alone jump more than one band over years.
        base = max(years_idx, min(title_idx, years_idx + 1))

    nudge = _signal_score(cp) if cp is not None else 0
    # Only nudge upward for very junior bands where richness is discriminating;
    # senior bands are set by title/years, not project counts.
    if base <= 2:
        base += nudge
    signals["nudge"] = nudge
    return SENIORITY_BANDS[_clamp(base)], signals


def classify_target_band(org_ctx: dict | None) -> Band | None:
    """The band of the ROLE being interviewed for, from the intake job role."""
    if not isinstance(org_ctx, dict):
        return None
    role = str(org_ctx.get("job_role") or "")
    idx = _band_from_title(role)
    if idx is None:
        return None
    return SENIORITY_BANDS[_clamp(idx)]


def build_calibration(profile: dict | None, org_ctx: dict | None, *,
                      cp=None, override: str | None = None) -> Calibration | None:
    """Assemble the full calibration. Never raises → None on any failure."""
    try:
        real, signals = classify_real_band(profile, cp, override)
        target = classify_target_band(org_ctx)
        oc = org_ctx if isinstance(org_ctx, dict) else {}
        role = str(oc.get("job_role") or "")
        jd = str(oc.get("job_description") or "")
        skills = None
        if isinstance(profile, dict):
            sk = profile.get("skills")
            if isinstance(sk, list):
                skills = [str(s) for s in sk]
        track = detect_track(role, skills)

        # People-management framing from the candidate's OWN title (their real
        # scope), not the target role — an IC interviewing for an EM role isn't
        # yet a manager. Skipped entirely on manual override.
        management = None
        if not signals.get("override"):
            cur_title = ""
            if isinstance(profile, dict):
                cur_title = str(profile.get("current_role") or profile.get("headline") or "")
            management = detect_management(cur_title)

        # Optional 4th-dimension / capability layers, each behind an enabling
        # default so a config author can turn them off without touching code.
        industry = None
        readiness = None
        try:
            from app.core.config_loader import cfg
            _cfg_live = getattr(cfg, "live", None)
        except Exception:  # noqa: BLE001
            _cfg_live = None
        if getattr(_cfg_live, "industry_context", True):
            industry = detect_industry(role, jd, skills)
        if getattr(_cfg_live, "capability_framing", True) and cp is not None:
            try:
                from app.live import career as _career
                readiness = _career.readiness_signal(cp)
            except Exception:  # noqa: BLE001
                readiness = None

        return Calibration(real_band=real, target_band=target, track=track,
                           overridden=bool(signals.get("override")),
                           management=management, industry=industry,
                           readiness=readiness, signals=signals)
    except Exception:  # noqa: BLE001
        return None


def calibration_directive(cal: Calibration | None) -> str:
    """Render the ANSWER GUIDANCE directive for the live answer prompt. Compact
    (a few lines), imperative, and truthful-by-construction. Empty on None."""
    if cal is None:
        return ""
    try:
        b = cal.real_band
        lines: list[str] = [
            f"SENIORITY CALIBRATION — pitch this answer at the level of a "
            f"{b.label}: someone whose scope is {b.focus}.",
            f"Use ownership language that fits: \"{b.ownership}\". "
            f"Foreground {b.emphasis}.",
            f"Do NOT overclaim {b.caution}. Stay strictly truthful to the "
            f"candidate's real experience — never invent titles, employers, "
            f"years, or scope beyond the resume.",
        ]
        if cal.management is not None:
            m = cal.management
            lines.append(
                f"This is a MANAGEMENT-track profile ({m.label}) — frame the answer around "
                f"leading {m.scope}: foreground {m.emphasis}. Lead with people leadership, "
                f"team and organisational outcomes, and delivery THROUGH others, not just "
                f"personal hands-on coding. Do NOT overclaim {m.caution}.")
        if cal.track is not None:
            lines.append(
                f"This is a {cal.track.label} interview — frame examples around "
                f"{cal.track.emphasis}.")
        if cal.industry is not None:
            lines.append(
                f"Industry context — this is a {cal.industry.label} role; where relevant, "
                f"ground examples in {cal.industry.hint}, without claiming domain-specific "
                f"years the resume does not show.")
        if cal.readiness:
            try:
                from app.live import career as _career
                cap = _career.capability_directive(cal.readiness)
                if cap:
                    lines.append(cap)
            except Exception:  # noqa: BLE001
                pass
        t = cal.target_band
        if t is not None and t.index > b.index:
            lines.append(
                f"The target role ({t.label}) sits above the candidate's current "
                f"level. Frame genuine, transferable strengths toward "
                f"{t.label}-level expectations ({t.emphasis}) and show a clear "
                f"growth trajectory — WITHOUT claiming experience, titles, or "
                f"years the candidate does not have.")
        elif t is not None and t.index < b.index:
            lines.append(
                f"The candidate exceeds the target role ({t.label}). Answer with "
                f"the depth and judgement of their real level while staying "
                f"relatable and not condescending.")
        # Always end on professionalism: even a fresher should sound polished.
        lines.append(
            "Whatever the level, deliver it with maximum professionalism: "
            "clear structure, confident and specific, concise, no filler.")
        return " ".join(lines)
    except Exception:  # noqa: BLE001
        return ""
