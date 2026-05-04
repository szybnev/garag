# Generation evaluation — `golden_set_v1`

Ran the full `QueryPipeline` over 50 queries from `data/golden/golden_set_v1.jsonl`. Generator: `zai-org/glm-4.7-flash`. Judge: `zai-org/glm-4.7-flash`.

> **Self-bias caveat.** This report uses `zai-org/glm-4.7-flash` as both generator and judge, which is known to inflate faithfulness by 5-15% vs a cross-model judge. Treat the absolute numbers as upper bounds; rely on per-category deltas and the manual 10-sample review for qualitative signal. A cross-model rerun (e.g. GPT-4o or Claude as judge) is deferred.

## Overall metrics

| metric | value | NFR target |
|---|---|---|
| format_rate | **1.000** | — |
| citation_acc (mechanical) | **1.000** | ≥ 0.85 |
| faithfulness (judge, /2 norm) | **0.900** | ≥ 0.80 |
| correctness (judge, /2 norm) | **0.920** | ≥ 0.70 |
| citation_support (judge, /2 norm) | **0.900** | — |
| judge format failures | 0 | — |
| mean / p95 latency | 1230 ms / 1993 ms | p95 ≤ 8000 ms |

## Per-category breakdown

| category | n | format | cit_acc | faith | corr | cit_sup | p95 ms |
|---|---|---|---|---|---|---|---|
| factual | 20 | 1.000 | 1.000 | 0.850 | 0.875 | 0.775 | 1911 |
| tool_usage | 15 | 1.000 | 1.000 | 0.867 | 0.900 | 0.967 | 1673 |
| multi_hop | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2469 |

## Manual review — 10 random samples

### Sample 1 — `g041` (tool_usage)

**Question:** How do you invoke the nuclei vulnerability scanner using a specific flag or pattern described in the snippet?

**Golden:** The snippet does not describe any flags, options, or invocation patterns for the nuclei command-line tool; it only displays contributor avatars.

**Candidate:** To invoke nuclei with a specific template, use the `-t` flag followed by the path to the YAML template file (e.g., `nuclei -u https://example.com -t /path/to/your-template.yaml`).

**Judge:** faith=1/2 corr=2/2 cit_sup=2/2

*Rationale:* The answer claims '-t' flag usage, which is supported by source [4]. However, the example URL 'https://example.com' is not explicitly in the context, making it partially unsupported (faithfulness=1). The answer matches the golden answer's substance (correctness=2). The citation [1] is loosely related to flags but not the specific '-t' usage, so citation_support=1.

### Sample 2 — `g008` (factual)

**Question:** What file extension do Microsoft-created administrative tools use when opened by MMC?

**Golden:** Microsoft-created administrative tools use the .msc file extension.

**Candidate:** Microsoft-created administrative tools use the .msc file extension when opened by MMC.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The answer matches the golden answer exactly. Context [1] explicitly states MMC can be used to open Microsoft created .msc files. The citation [1] is on-point and directly supports the claim.

### Sample 3 — `g002` (factual)

**Question:** Which military unit is the Sandworm Team attributed to?

**Golden:** The Sandworm Team is attributed to Russia's General Staff Main Intelligence Directorate (GRU) Main Center for Special Technologies (GTsST) military unit 74455.

**Candidate:** The Sandworm Team is attributed to Russia's General Staff Main Intelligence Directorate (GRU) Main Center for Special Technologies (GTsST) military unit 74455.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The answer matches the golden answer exactly. The citation [1] directly supports the attribution to GRU GTsST military unit 74455.

### Sample 4 — `g018` (tool_usage)

**Question:** How do you enable OS detection in nmap and view the target's uptime guess?

**Golden:** Use the -O flag to enable OS detection, and add the -v flag for verbose mode to see the uptime guess.

**Candidate:** To enable OS detection, use the -O option. To view the target's uptime guess, enable verbose mode (-v) along with -O, which reports the guess based on the TCP timestamp option (RFC 1323).

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* All claims (-O, -v, TCP timestamp RFC 1323) are directly supported by context [1]. The answer is semantically equivalent to the golden answer and citations are on-point.

### Sample 5 — `g016` (tool_usage)

**Question:** How can you use the -sO flag in Nmap to determine which IP protocols are supported by a target machine?

**Golden:** The -sO flag performs an IP protocol scan that cycles through IP protocol numbers rather than port numbers to identify supported protocols like TCP or ICMP.

**Candidate:** Use the -sO flag to determine which IP protocols (TCP, ICMP, IGMP, etc.) are supported by a target machine. This scan type cycles through IP protocol numbers rather than TCP or UDP port numbers, and the -p option specifies the protocol numbers you wish to scan for (0–255).

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* All claims are directly supported by context [1] and [2]. The answer accurately describes -sO and -p usage.

### Sample 6 — `g015` (tool_usage)

**Question:** How do you perform a protocol scan using nmap, and which specific IP protocol fields are iterated through during this process?

**Golden:** A protocol scan is performed by sending IP packet headers and iterating through the eight-bit IP protocol field. The exceptions where headers contain data include TCP, UDP, ICMP, SCTP, and IGMP.

**Candidate:** To perform a protocol scan using nmap, use the -sO option, which cycles through IP protocol numbers rather than TCP or UDP port numbers. The scan iterates through the eight-bit IP protocol field of IP packet headers.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* Both claims (-sO and iterating the 8-bit IP protocol field) are directly supported by source [1]. The answer matches the golden answer exactly.

### Sample 7 — `g009` (factual)

**Question:** What API is used for high-level communication with XPC services?

**Golden:** The NSXPCConnection API is used for high-level communication with XPC services.

**Candidate:** Applications can send messages to the XPC Service daemon using the high-level NSXPCConnection API.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* Candidate answer matches golden exactly. Context [1] explicitly states 'Applications can send messages to the XPC Service daemon... using the high level NSXPCConnection API'. All claims are supported and cited correctly.

### Sample 8 — `g007` (factual)

**Question:** Which specific API endpoints can adversaries query to enumerate email addresses in Office 365 environments?

**Golden:** Adversaries may query publicly available API endpoints such as autodiscover and GetCredentialType.

**Candidate:** Adversaries may enumerate email addresses in Office 365 environments by querying publicly available API endpoints such as autodiscover and GetCredentialType.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The answer matches the golden answer exactly and cites source [1] which explicitly lists 'autodiscover' and 'GetCredentialType' as API endpoints used to enumerate email addresses in Office 365.

### Sample 9 — `g035` (factual)

**Question:** What is the purpose of dynamic resolution in the context of command and control infrastructure?

**Golden:** Adversaries use dynamic resolution to evade common detections and remediations or to reestablish command and control when contact with the primary server is lost.

**Candidate:** Adversaries use dynamic resolution to evade detection and remediation by dynamically adjusting parameters like domain names, IP addresses, or port numbers for command and control. This technique is also used for fallback channels to reestablish command and control when contact is lost with the primary server.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* All claims match context. Citations [1] support evasion and parameter adjustment; [1] supports fallback channels. No contradictions.

### Sample 10 — `g006` (factual)

**Question:** Which platforms are associated with the Hardware technique in the provided snippet?

**Golden:** The platforms associated with the Hardware technique are PRE.

**Candidate:** The provided sources do not contain enough information to answer this question.

**Judge:** faith=1/2 corr=1/2 cit_sup=1/2

*Rationale:* Answer claims info is missing, but source [10] explicitly lists PRE. Partially supported by citation [1] which lists platforms but doesn't specify Hardware technique.
