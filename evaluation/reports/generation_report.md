# Generation evaluation — `golden_set_v1`

Ran the full `QueryPipeline` over 50 queries from `data/golden/golden_set_v1.jsonl`. Generator: `qwen3.5:35b`. Judge: `qwen3.5:35b`.

> **Self-bias caveat.** This report uses `qwen3.5:35b` as both generator and judge, which is known to inflate faithfulness by 5-15% vs a cross-model judge. Treat the absolute numbers as upper bounds; rely on per-category deltas and the manual 10-sample review for qualitative signal. A cross-model rerun (e.g. GPT-4o or Claude as judge) is deferred to PoxekBook increment 2.

## Overall metrics

| metric | value | NFR target |
|---|---|---|
| format_rate | **1.000** | — |
| citation_acc (mechanical) | **1.000** | ≥ 0.85 |
| faithfulness (judge, /2 norm) | **0.930** | ≥ 0.80 |
| correctness (judge, /2 norm) | **0.900** | ≥ 0.70 |
| citation_support (judge, /2 norm) | **0.960** | — |
| judge format failures | 0 | — |
| mean / p95 latency | 2274 ms / 3756 ms | p95 ≤ 8000 ms |

## Per-category breakdown

| category | n | format | cit_acc | faith | corr | cit_sup | p95 ms |
|---|---|---|---|---|---|---|---|
| factual | 20 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 | 5346 |
| tool_usage | 15 | 1.000 | 1.000 | 0.833 | 0.800 | 0.900 | 3756 |
| multi_hop | 15 | 1.000 | 1.000 | 0.933 | 0.933 | 0.967 | 4322 |

## Manual review — 10 random samples

### Sample 1 — `g041` (tool_usage)

**Question:** How do you invoke the nuclei vulnerability scanner using a specific flag or pattern described in the snippet?

**Golden:** The snippet does not describe any flags, options, or invocation patterns for the nuclei command-line tool; it only displays contributor avatars.

**Candidate:** The provided sources do not contain information on invoking the Nuclei vulnerability scanner with a specific flag or pattern described in the snippet, as the snippet regarding specific flags (like --scanflags) pertains to the nmap tool, not nuclei. While the Nuclei documentation shows usage of flags like -u, -target, and -t for templates, no specific pattern invocation beyond these standard flags is detailed in the provided context.

**Judge:** faith=1/2 corr=1/2 cit_sup=2/2

*Rationale:* Faithfulness is 1 because the candidate claims the snippet describes flags for 'nmap' (which is true for the specific --scanflags flag mentioned), but the question asks about 'nuclei'. The candidate correctly identifies that the snippet does not describe a specific nuclei pattern, but adds external knowledge about nmap flags not present in the specific 'nuclei' context chunks. Correctness is 1 because it correctly states the snippet lacks the specific nuclei pattern, though it introduces nmap details not in the reference. Citation support is 2 as the citations directly support the claims about nuclei flags and the nmap --scanflags flag.

### Sample 2 — `g008` (factual)

**Question:** What file extension do Microsoft-created administrative tools use when opened by MMC?

**Golden:** Microsoft-created administrative tools use the .msc file extension.

**Candidate:** Microsoft-created administrative tools opened by MMC use the .msc file extension. These files contain custom consoles or snap-ins used to manage Windows systems locally or remotely.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer correctly identifies the .msc extension and the role of snap-ins, which is fully supported by the retrieval context (specifically chunk [1]). The citations provided directly back the quoted claims regarding MMC opening .msc files and containing Microsoft-created snap-ins. The answer is semantically equivalent to the reference.

### Sample 3 — `g002` (factual)

**Question:** Which military unit is the Sandworm Team attributed to?

**Golden:** The Sandworm Team is attributed to Russia's General Staff Main Intelligence Directorate (GRU) Main Center for Special Technologies (GTsST) military unit 74455.

**Candidate:** The Sandworm Team is attributed to Russia's General Staff Main Intelligence Directorate (GRU) Main Center for Special Technologies (GTsST) military unit 74455.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer is a verbatim match to the reference answer and is fully supported by the provided context (specifically chunk [1]). The citation provided directly quotes the exact claim made in the answer, satisfying all rubric criteria for perfect faithfulness, correctness, and citation support.

### Sample 4 — `g018` (tool_usage)

**Question:** How do you enable OS detection in nmap and view the target's uptime guess?

**Golden:** Use the -O flag to enable OS detection, and add the -v flag for verbose mode to see the uptime guess.

**Candidate:** OS detection is enabled using the -O flag, which also allows Nmap to guess a target's uptime using the TCP timestamp option. However, this uptime guess is only printed in verbose mode (-v), as it can be inaccurate due to timestamp counter issues.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer correctly identifies the -O flag for OS detection and the -v flag for viewing uptime, matching the reference. It accurately reflects the context regarding TCP timestamps and the condition that uptime is printed only in verbose mode. Citations [1] and [2] directly support the claims about the mechanism and the flag requirements.

### Sample 5 — `g016` (tool_usage)

**Question:** How can you use the -sO flag in Nmap to determine which IP protocols are supported by a target machine?

**Golden:** The -sO flag performs an IP protocol scan that cycles through IP protocol numbers rather than port numbers to identify supported protocols like TCP or ICMP.

**Candidate:** The -sO flag in Nmap performs an IP protocol scan that cycles through IP protocol numbers (0–255) rather than TCP or UDP port numbers to determine which IP protocols (such as TCP, ICMP, and IGMP) are supported by the target machine. This scan type uses the -p option to specify which protocol numbers to scan, and the results are reported within the normal port table.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer accurately reflects the reference answer and the provided context. It correctly explains the -sO flag's function, the range of protocol numbers (0-255), and the use of the -p option, all of which are explicitly supported by the retrieved chunks. The citations directly back the specific claims made in the answer.

### Sample 6 — `g015` (tool_usage)

**Question:** How do you perform a protocol scan using nmap, and which specific IP protocol fields are iterated through during this process?

**Golden:** A protocol scan is performed by sending IP packet headers and iterating through the eight-bit IP protocol field. The exceptions where headers contain data include TCP, UDP, ICMP, SCTP, and IGMP.

**Candidate:** To perform a protocol scan using Nmap, you use the -sO option, which sends IP packet headers and iterates through the eight-bit IP protocol field (values 0–255) rather than port numbers. This scan determines which IP protocols (such as TCP, ICMP, IGMP, SCTP, and UDP) are supported by target machines. The specific protocol numbers to scan can be specified using the -p option with values ranging from 0 to 255.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer accurately describes the -sO flag, the iteration of the 8-bit IP protocol field (0-255), and the specific protocols (TCP, ICMP, IGMP, SCTP, UDP) mentioned in the context. All claims are directly supported by the provided citations, which confirm the mechanism, the range, and the list of exception protocols.

### Sample 7 — `g009` (factual)

**Question:** What API is used for high-level communication with XPC services?

**Golden:** The NSXPCConnection API is used for high-level communication with XPC services.

**Candidate:** The high-level API used for communication with XPC services is the NSXPCConnection API. Applications use this API to send messages to the XPC Service daemon to handle tasks requiring elevated privileges.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer correctly identifies NSXPCConnection as the high-level API, matching the reference. The context explicitly states this API is used for high-level communication with XPC services. The candidate's additional details about sending messages to the daemon are also supported by the context. The citation directly supports the quoted claims.

### Sample 8 — `g007` (factual)

**Question:** Which specific API endpoints can adversaries query to enumerate email addresses in Office 365 environments?

**Golden:** Adversaries may query publicly available API endpoints such as autodiscover and GetCredentialType.

**Candidate:** Adversaries can enumerate email addresses in Office 365 environments by querying the autodiscover and GetUserRealm API endpoints, as well as the GetCredentialType endpoint. These publicly available endpoints allow probing and analysis of authentication services to reveal valid usernames.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer correctly identifies autodiscover, GetUserRealm, and GetCredentialType as endpoints for enumerating email addresses in Office 365. This is fully supported by the retrieval context: chunk [1] lists autodiscover and GetCredentialType, while chunk [3] lists GetUserRealm and autodiscover. The candidate synthesizes these sources accurately without contradiction.

### Sample 9 — `g035` (factual)

**Question:** What is the purpose of dynamic resolution in the context of command and control infrastructure?

**Golden:** Adversaries use dynamic resolution to evade common detections and remediations or to reestablish command and control when contact with the primary server is lost.

**Candidate:** In the context of command and control infrastructure, dynamic resolution is used by adversaries to dynamically establish connections to evade common detections and remediations by adjusting parameters such as domain names, IP addresses, or port numbers. It also serves as a fallback mechanism to reestablish command and control if contact with the primary server is lost. Additionally, it makes it harder for defenders to block, track, or take over the command and control channel by dynamically identifying destination domains rather than relying on static lists.

**Judge:** faith=2/2 corr=2/2 cit_sup=2/2

*Rationale:* The candidate answer accurately reflects the reference answer's core points (evasion and fallback). It also correctly incorporates specific details from the retrieval context (adjusting parameters, DGAs, difficulty for defenders). All claims are directly supported by the provided chunks, and the citations match the quoted content perfectly.

### Sample 10 — `g006` (factual)

**Question:** Which platforms are associated with the Hardware technique in the provided snippet?

**Golden:** The platforms associated with the Hardware technique are PRE.

**Candidate:** The provided sources do not contain information about a technique named 'Hardware'; they list platforms for techniques T1049, T1562.006, T1072, T1518.002, and T1552.004, none of which are identified as 'Hardware'.

**Judge:** faith=2/2 corr=0/2 cit_sup=2/2

*Rationale:* The candidate answer is factually supported by the provided context, which lists platforms for specific MITRE techniques but contains no mention of a 'Hardware' technique. However, it is incorrect relative to the reference answer, which explicitly states 'PRE' is the platform for the Hardware technique. The candidate fails to identify the correct answer.
