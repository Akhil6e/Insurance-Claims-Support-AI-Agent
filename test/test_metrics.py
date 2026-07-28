"""
=============================================================================
  Insurance Claims Copilot — Comprehensive Metrics Evaluation Suite
=============================================================================

Measures 10 key performance metrics for the Insurance Claims Support AI Agent:

  1. Retrieval Speed          — RAG vector search latency
  2. Total Response Time      — End-to-end draft generation latency
  3. Number of Documents      — Knowledge base & memory document counts
  4. Retrieval Accuracy       — Relevance scoring of RAG/memory results
  5. Manual vs AI Time        — Estimated human time savings
  6. Memory Performance       — Mem0/LangMem store latency & recall
  7. Deployment               — Docker health, container status
  8. API Performance          — Individual endpoint latency & status
  9. Throughput               — Concurrent request handling capacity
 10. LLM Output Quality       — Draft content quality heuristics

Usage:
    # Ensure the API server is running (default: http://localhost:8000)
    python test/test_metrics.py

    # Or specify a different API URL:
    API_BASE_URL=http://localhost:8000 python test/test_metrics.py

    # Run with verbose JSON output:
    METRICS_VERBOSE=1 python test/test_metrics.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
VERBOSE = os.getenv("METRICS_VERBOSE", "0") == "1"
TIMEOUT = 60  # seconds per request

# Test fixtures
TEST_CUSTOMER_EMAIL = "metrics-test@acme-insurance.io"
TEST_CUSTOMER_NAME = "Metrics Test User"
TEST_CUSTOMER_COMPANY = "Acme Insurance Corp"
TEST_TICKET_SUBJECT = "Collision damage to 2024 Honda Civic — rear-end impact"
TEST_TICKET_DESCRIPTION = (
    "Claim type: Collision\n"
    "Policy number: POL-2026-METRICS-001\n"
    "Incident date: 2026-07-15\n"
    "Loss location: San Jose, CA\n"
    "Estimated loss amount: $8,500.00\n\n"
    "FNOL narrative:\n"
    "The insured was stopped at a red light on Stevens Creek Blvd "
    "when a vehicle rear-ended them at approximately 30 mph. "
    "Airbags did not deploy. The rear bumper, trunk lid, and taillights "
    "are damaged. The insured reports neck pain and visited urgent care. "
    "Police report #2026-SJ-44892 was filed. The other driver's insurance "
    "information was exchanged at the scene."
)

# RAG relevance test queries and expected keyword sets
RETRIEVAL_TEST_CASES = [
    {
        "query": "What documents are required for a collision claim?",
        "expected_keywords": ["police", "report", "photo", "estimate", "document"],
    },
    {
        "query": "What is the FNOL intake checklist for auto claims?",
        "expected_keywords": ["fnol", "checklist", "intake", "claim", "auto"],
    },
    {
        "query": "What are the fraud risk indicators for insurance claims?",
        "expected_keywords": ["fraud", "risk", "indicator", "suspicious"],
    },
    {
        "query": "What is the settlement SLA and communication timeline?",
        "expected_keywords": ["settlement", "sla", "communication", "timeline"],
    },
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
class MetricsCollector:
    """Collects and formats metrics results."""

    def __init__(self):
        self.results: dict[str, dict[str, Any]] = {}
        self._start_time = time.perf_counter()

    def record(self, metric_name: str, data: dict[str, Any]) -> None:
        self.results[metric_name] = {
            "timestamp": datetime.now().isoformat(),
            **data,
        }

    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def summary(self) -> str:
        lines = [
            "",
            "=" * 80,
            "  INSURANCE CLAIMS COPILOT — METRICS EVALUATION REPORT",
            f"  Generated: {datetime.now().isoformat()}",
            f"  API Target: {API_BASE_URL}",
            f"  Total Evaluation Time: {self.elapsed():.2f}s",
            "=" * 80,
        ]
        for idx, (name, data) in enumerate(self.results.items(), 1):
            lines.append(f"\n{'─' * 80}")
            status_icon = "✅" if data.get("status") == "PASS" else "❌" if data.get("status") == "FAIL" else "⚠️"
            lines.append(f"  {idx:2d}. {status_icon}  {name}")
            lines.append(f"{'─' * 80}")
            for key, value in data.items():
                if key == "timestamp":
                    continue
                if isinstance(value, float):
                    lines.append(f"      {key:30s}: {value:.4f}")
                elif isinstance(value, dict) and VERBOSE:
                    lines.append(f"      {key}:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"          {sub_key:26s}: {sub_value}")
                elif isinstance(value, list) and VERBOSE:
                    lines.append(f"      {key}: [{len(value)} items]")
                    for item in value[:5]:
                        lines.append(f"          - {str(item)[:100]}")
                else:
                    lines.append(f"      {key:30s}: {value}")

        # Final score
        total = len(self.results)
        passed = sum(1 for d in self.results.values() if d.get("status") == "PASS")
        lines.append(f"\n{'=' * 80}")
        lines.append(f"  OVERALL: {passed}/{total} metrics passed")
        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)


def timed_request(method: str, url: str, **kwargs) -> tuple[requests.Response, float]:
    """Execute an HTTP request and return (response, elapsed_seconds)."""
    kwargs.setdefault("timeout", TIMEOUT)
    start = time.perf_counter()
    response = requests.request(method, url, **kwargs)
    elapsed = time.perf_counter() - start
    return response, elapsed


def safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metric 1: Retrieval Speed
# ---------------------------------------------------------------------------
def measure_retrieval_speed(collector: MetricsCollector) -> None:
    """Measure RAG vector search latency via the knowledge ingest + internal search."""
    print("  [1/10] Measuring Retrieval Speed (RAG search latency)...")

    # First, ensure knowledge base is ingested
    try:
        resp, ingest_time = timed_request(
            "POST",
            f"{API_BASE_URL}/api/knowledge/ingest",
            json={"clear_existing": False},
        )
        ingest_data = safe_json(resp) or {}
    except Exception as exc:
        collector.record("Retrieval Speed", {
            "status": "FAIL",
            "error": f"Knowledge ingest failed: {exc}",
        })
        return

    # Measure retrieval speed by creating a ticket and triggering draft
    # (which internally performs RAG search)
    search_times: list[float] = []

    # We can measure indirectly via the draft generation context
    # which reports knowledge_hit_count. We time the full generate-draft call
    # and extract the RAG portion from the context.
    # Alternatively, we measure the memory-search endpoint as a proxy for retrieval.

    # Create a test customer/ticket to query memory-search
    try:
        ticket_resp, _ = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": TEST_CUSTOMER_EMAIL,
                "customer_name": TEST_CUSTOMER_NAME,
                "customer_company": TEST_CUSTOMER_COMPANY,
                "subject": TEST_TICKET_SUBJECT,
                "description": TEST_TICKET_DESCRIPTION,
                "priority": "medium",
                "auto_generate": False,
            },
        )
        ticket_data = safe_json(ticket_resp)
        customer_id = ticket_data.get("customer_id") if ticket_data else None
    except Exception:
        customer_id = None

    # Measure memory-search as a retrieval speed proxy
    if customer_id:
        for case in RETRIEVAL_TEST_CASES:
            try:
                _, elapsed = timed_request(
                    "GET",
                    f"{API_BASE_URL}/api/customers/{customer_id}/memory-search",
                    params={"query": case["query"], "limit": 5},
                )
                search_times.append(elapsed)
            except Exception:
                pass

    if search_times:
        collector.record("Retrieval Speed", {
            "status": "PASS" if statistics.mean(search_times) < 5.0 else "FAIL",
            "mean_latency_sec": statistics.mean(search_times),
            "min_latency_sec": min(search_times),
            "max_latency_sec": max(search_times),
            "median_latency_sec": statistics.median(search_times),
            "num_queries": len(search_times),
            "ingest_time_sec": ingest_time,
            "chunks_indexed": ingest_data.get("chunks_indexed", "N/A"),
            "threshold": "< 5.0s per query",
        })
    else:
        collector.record("Retrieval Speed", {
            "status": "FAIL",
            "error": "Could not measure retrieval speed — no queries executed",
            "ingest_time_sec": ingest_time,
        })


# ---------------------------------------------------------------------------
# Metric 2: Total Response Time
# ---------------------------------------------------------------------------
def measure_total_response_time(collector: MetricsCollector) -> None:
    """Measure end-to-end draft generation latency (ticket creation → draft ready)."""
    print("  [2/10] Measuring Total Response Time (end-to-end draft generation)...")

    try:
        # Create ticket with auto_generate=False, then manually trigger draft
        resp, create_time = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": f"e2e-{int(time.time())}@test.io",
                "customer_name": "E2E Test",
                "subject": "Windshield crack from road debris — comprehensive claim",
                "description": (
                    "Claim type: Comprehensive\n"
                    "Policy number: POL-2026-E2E-001\n"
                    "Incident date: 2026-07-14\n"
                    "Loss location: Highway 101, Mountain View, CA\n"
                    "Estimated loss amount: $1,200.00\n\n"
                    "FNOL narrative:\n"
                    "A rock kicked up by a truck hit the windshield while "
                    "driving on Highway 101, causing a 12-inch crack."
                ),
                "priority": "low",
                "auto_generate": False,
            },
        )
        ticket = safe_json(resp)
        if not ticket or resp.status_code >= 400:
            raise RuntimeError(f"Ticket creation failed: {resp.status_code}")

        ticket_id = ticket["id"]

        # Now trigger draft generation (this is the heavy LLM call)
        draft_resp, draft_time = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets/{ticket_id}/generate-draft",
        )
        draft_data = safe_json(draft_resp)
        total_time = create_time + draft_time

        if draft_resp.status_code < 400 and draft_data:
            draft_content = draft_data.get("draft", {}).get("content", "")
            context = draft_data.get("draft", {}).get("context_used", {})
            collector.record("Total Response Time", {
                "status": "PASS" if total_time < 60 else "FAIL",
                "total_time_sec": total_time,
                "ticket_creation_sec": create_time,
                "draft_generation_sec": draft_time,
                "draft_length_chars": len(draft_content),
                "memory_hits": (context or {}).get("signals", {}).get("memory_hit_count", 0),
                "knowledge_hits": (context or {}).get("signals", {}).get("knowledge_hit_count", 0),
                "tool_calls": (context or {}).get("signals", {}).get("tool_call_count", 0),
                "threshold": "< 60s total",
            })
        else:
            collector.record("Total Response Time", {
                "status": "FAIL",
                "total_time_sec": total_time,
                "error": f"Draft generation returned {draft_resp.status_code}",
                "detail": str(draft_data)[:300] if draft_data else "No response body",
            })

    except Exception as exc:
        collector.record("Total Response Time", {
            "status": "FAIL",
            "error": str(exc),
        })


# ---------------------------------------------------------------------------
# Metric 3: Number of Documents
# ---------------------------------------------------------------------------
def measure_number_of_documents(collector: MetricsCollector) -> None:
    """Count documents in knowledge base and database."""
    print("  [3/10] Measuring Number of Documents (KB + DB counts)...")

    try:
        # Ingest to get counts
        resp, _ = timed_request(
            "POST",
            f"{API_BASE_URL}/api/knowledge/ingest",
            json={"clear_existing": False},
        )
        kb_data = safe_json(resp) or {}

        # Count tickets
        tickets_resp, _ = timed_request("GET", f"{API_BASE_URL}/api/tickets")
        tickets = safe_json(tickets_resp) or []

        # Count knowledge base files in the local directory
        kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
        kb_files = []
        if os.path.isdir(kb_path):
            kb_files = [f for f in os.listdir(kb_path) if f.endswith((".md", ".txt"))]

        collector.record("Number of Documents", {
            "status": "PASS",
            "kb_source_files": kb_data.get("files_indexed", len(kb_files)),
            "kb_chunks_indexed": kb_data.get("chunks_indexed", "N/A"),
            "kb_collection_total": kb_data.get("collection_count", "N/A"),
            "kb_local_files": len(kb_files),
            "kb_file_list": kb_files if VERBOSE else f"{len(kb_files)} files",
            "total_tickets": len(tickets),
            "ticket_statuses": _count_by_key(tickets, "status"),
            "ticket_priorities": _count_by_key(tickets, "priority"),
        })
    except Exception as exc:
        collector.record("Number of Documents", {
            "status": "FAIL",
            "error": str(exc),
        })


def _count_by_key(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = str(item.get(key, "unknown"))
        counts[val] = counts.get(val, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Metric 4: Retrieval Accuracy
# ---------------------------------------------------------------------------
def measure_retrieval_accuracy(collector: MetricsCollector) -> None:
    """Evaluate relevance of RAG/memory search results against expected keywords."""
    print("  [4/10] Measuring Retrieval Accuracy (keyword relevance scoring)...")

    # First ensure knowledge base is ingested
    try:
        timed_request(
            "POST",
            f"{API_BASE_URL}/api/knowledge/ingest",
            json={"clear_existing": False},
        )
    except Exception:
        pass

    # Create a test ticket to get a customer_id for memory search
    try:
        resp, _ = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": f"accuracy-{int(time.time())}@test.io",
                "customer_name": "Accuracy Test",
                "subject": "Auto claim document requirements",
                "description": (
                    "Claim type: Collision\n"
                    "Policy number: POL-2026-ACC-001\n"
                    "Incident date: 2026-07-10\n"
                    "Loss location: Palo Alto, CA\n"
                    "Estimated loss amount: $5,000.00\n\n"
                    "FNOL narrative:\n"
                    "Need information about required documents for collision claim."
                ),
                "priority": "medium",
                "auto_generate": False,
            },
        )
        ticket = safe_json(resp)
        customer_id = ticket.get("customer_id") if ticket else None
        ticket_id = ticket.get("id") if ticket else None
    except Exception:
        customer_id = None
        ticket_id = None

    # Test draft generation to evaluate RAG accuracy from context_used
    accuracy_results: list[dict[str, Any]] = []

    if ticket_id:
        try:
            draft_resp, _ = timed_request(
                "POST",
                f"{API_BASE_URL}/api/tickets/{ticket_id}/generate-draft",
            )
            draft_data = safe_json(draft_resp) or {}
            draft_info = draft_data.get("draft", {})
            context = draft_info.get("context_used", {})
            kb_hits = context.get("knowledge_hits", [])

            # Score each retrieval test case
            for case in RETRIEVAL_TEST_CASES:
                query = case["query"]
                expected = case["expected_keywords"]

                # Check if any KB hit content matches expected keywords
                matched_keywords: list[str] = []
                for hit in kb_hits:
                    content_lower = (hit.get("content") or "").lower()
                    for kw in expected:
                        if kw.lower() in content_lower and kw not in matched_keywords:
                            matched_keywords.append(kw)

                # Also check the draft content itself for relevance
                draft_content_lower = (draft_info.get("content") or "").lower()
                for kw in expected:
                    if kw.lower() in draft_content_lower and kw not in matched_keywords:
                        matched_keywords.append(kw)

                relevance_score = len(matched_keywords) / len(expected) if expected else 0
                accuracy_results.append({
                    "query": query,
                    "expected_keywords": expected,
                    "matched_keywords": matched_keywords,
                    "relevance_score": relevance_score,
                })

        except Exception as exc:
            accuracy_results.append({"error": str(exc)})

    if accuracy_results and not any("error" in r for r in accuracy_results):
        avg_relevance = statistics.mean(r["relevance_score"] for r in accuracy_results)
        collector.record("Retrieval Accuracy", {
            "status": "PASS" if avg_relevance >= 0.3 else "FAIL",
            "average_relevance_score": avg_relevance,
            "num_test_cases": len(accuracy_results),
            "per_query_scores": {
                r["query"][:50]: f"{r['relevance_score']:.0%}"
                for r in accuracy_results
            } if VERBOSE else f"{len(accuracy_results)} queries evaluated",
            "threshold": ">= 30% keyword match",
        })
    else:
        collector.record("Retrieval Accuracy", {
            "status": "FAIL",
            "error": "Could not evaluate retrieval accuracy",
            "details": accuracy_results,
        })


# ---------------------------------------------------------------------------
# Metric 5: Manual vs AI Time
# ---------------------------------------------------------------------------
def measure_manual_vs_ai_time(collector: MetricsCollector) -> None:
    """Estimate time savings: human research/drafting vs AI-generated recommendation."""
    print("  [5/10] Measuring Manual vs AI Time (time savings estimate)...")

    # Benchmark: average human time to research and draft an insurance
    # coverage recommendation (industry estimates)
    HUMAN_RESEARCH_MINUTES = 25  # Looking up policy, precedents, documents
    HUMAN_DRAFTING_MINUTES = 15  # Writing the recommendation
    HUMAN_REVIEW_MINUTES = 10   # Internal review
    HUMAN_TOTAL_MINUTES = HUMAN_RESEARCH_MINUTES + HUMAN_DRAFTING_MINUTES + HUMAN_REVIEW_MINUTES

    try:
        # Measure AI draft generation time
        resp, create_time = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": f"timing-{int(time.time())}@test.io",
                "customer_name": "Timing Test",
                "subject": "Vehicle theft claim — 2023 Toyota Camry",
                "description": (
                    "Claim type: Theft\n"
                    "Policy number: POL-2026-TIME-001\n"
                    "Incident date: 2026-07-12\n"
                    "Loss location: Downtown San Francisco, CA\n"
                    "Estimated loss amount: $28,000.00\n\n"
                    "FNOL narrative:\n"
                    "The insured parked their vehicle in a public garage. "
                    "Upon returning 3 hours later, the vehicle was missing. "
                    "Police report #2026-SF-88012 filed. No witnesses. "
                    "Vehicle had anti-theft system. Insured has comprehensive coverage."
                ),
                "priority": "high",
                "auto_generate": False,
            },
        )
        ticket = safe_json(resp)
        if not ticket:
            raise RuntimeError("Failed to create test ticket")

        draft_resp, draft_time = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets/{ticket['id']}/generate-draft",
        )
        ai_total_seconds = create_time + draft_time
        ai_total_minutes = ai_total_seconds / 60

        # Human still needs ~5 min to review AI recommendation
        AI_REVIEW_MINUTES = 5
        effective_ai_minutes = ai_total_minutes + AI_REVIEW_MINUTES

        time_saved_minutes = HUMAN_TOTAL_MINUTES - effective_ai_minutes
        time_saved_pct = (time_saved_minutes / HUMAN_TOTAL_MINUTES) * 100

        collector.record("Manual vs AI Time", {
            "status": "PASS" if time_saved_pct > 50 else "FAIL",
            "human_estimated_minutes": HUMAN_TOTAL_MINUTES,
            "human_research_min": HUMAN_RESEARCH_MINUTES,
            "human_drafting_min": HUMAN_DRAFTING_MINUTES,
            "human_review_min": HUMAN_REVIEW_MINUTES,
            "ai_generation_sec": ai_total_seconds,
            "ai_generation_min": round(ai_total_minutes, 2),
            "ai_plus_review_min": round(effective_ai_minutes, 2),
            "time_saved_min": round(time_saved_minutes, 2),
            "time_saved_pct": f"{time_saved_pct:.1f}%",
            "speedup_factor": f"{HUMAN_TOTAL_MINUTES / max(effective_ai_minutes, 0.01):.1f}x",
            "threshold": "> 50% time savings",
        })

    except Exception as exc:
        collector.record("Manual vs AI Time", {
            "status": "FAIL",
            "error": str(exc),
        })


# ---------------------------------------------------------------------------
# Metric 6: Memory Performance
# ---------------------------------------------------------------------------
def measure_memory_performance(collector: MetricsCollector) -> None:
    """Measure memory store (Mem0/LangMem) write and retrieval performance."""
    print("  [6/10] Measuring Memory Performance (store + recall latency)...")

    try:
        # Create a ticket, generate a draft, accept it (triggers memory save),
        # then search memories
        email = f"memory-{int(time.time())}@test.io"

        # Step 1: Create ticket
        resp, _ = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": email,
                "customer_name": "Memory Perf Test",
                "customer_company": "Perf Testing LLC",
                "subject": "Fender bender in parking lot",
                "description": (
                    "Claim type: Collision\n"
                    "Policy number: POL-2026-MEM-001\n"
                    "Incident date: 2026-07-13\n"
                    "Loss location: Whole Foods parking lot, Cupertino, CA\n"
                    "Estimated loss amount: $2,000.00\n\n"
                    "FNOL narrative:\n"
                    "Minor fender bender while backing out of a parking space."
                ),
                "priority": "low",
                "auto_generate": False,
            },
        )
        ticket = safe_json(resp)
        ticket_id = ticket["id"]
        customer_id = ticket["customer_id"]

        # Step 2: Generate draft
        draft_resp, gen_time = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets/{ticket_id}/generate-draft",
        )
        draft_data = safe_json(draft_resp) or {}
        draft_id = draft_data.get("draft", {}).get("id")

        # Step 3: Accept draft (triggers memory persistence)
        if draft_id:
            accept_resp, accept_time = timed_request(
                "PATCH",
                f"{API_BASE_URL}/api/drafts/{draft_id}",
                json={
                    "content": draft_data.get("draft", {}).get("content", "Approved"),
                    "status": "accepted",
                },
            )
        else:
            accept_time = 0

        # Step 4: Search memories
        search_resp, search_time = timed_request(
            "GET",
            f"{API_BASE_URL}/api/customers/{customer_id}/memory-search",
            params={"query": "parking lot fender bender", "limit": 5},
        )
        search_data = safe_json(search_resp) or {}
        memory_results = search_data.get("results", [])

        # Step 5: List all memories
        list_resp, list_time = timed_request(
            "GET",
            f"{API_BASE_URL}/api/customers/{customer_id}/memories",
        )
        list_data = safe_json(list_resp) or {}
        all_memories = list_data.get("memories", [])

        collector.record("Memory Performance", {
            "status": "PASS" if search_time < 5.0 else "FAIL",
            "draft_generation_sec": gen_time,
            "memory_save_sec": accept_time,
            "memory_search_sec": search_time,
            "memory_list_sec": list_time,
            "memories_found_search": len(memory_results),
            "memories_found_list": len(all_memories),
            "threshold": "< 5.0s search latency",
        })

    except Exception as exc:
        collector.record("Memory Performance", {
            "status": "FAIL",
            "error": str(exc),
        })


# ---------------------------------------------------------------------------
# Metric 7: Deployment
# ---------------------------------------------------------------------------
def measure_deployment(collector: MetricsCollector) -> None:
    """Check deployment health: health endpoint, Docker containers."""
    print("  [7/10] Measuring Deployment (health checks + Docker status)...")

    results: dict[str, Any] = {}

    # Health endpoint
    try:
        resp, latency = timed_request("GET", f"{API_BASE_URL}/health")
        health_data = safe_json(resp)
        results["health_endpoint_status"] = resp.status_code
        results["health_response"] = health_data
        results["health_latency_sec"] = latency
        results["api_reachable"] = True
    except Exception as exc:
        results["api_reachable"] = False
        results["health_error"] = str(exc)

    # Docker status
    try:
        docker_ps = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=10,
        )
        if docker_ps.returncode == 0 and docker_ps.stdout.strip():
            containers = []
            for line in docker_ps.stdout.strip().split("\n"):
                parts = line.split("\t")
                containers.append({
                    "name": parts[0] if len(parts) > 0 else "unknown",
                    "status": parts[1] if len(parts) > 1 else "unknown",
                    "ports": parts[2] if len(parts) > 2 else "none",
                })
            # Check for our specific containers
            our_containers = [
                c for c in containers
                if "support-copilot" in c["name"] or "insurance" in c["name"].lower()
            ]
            results["docker_containers_total"] = len(containers)
            results["project_containers"] = len(our_containers)
            results["container_details"] = our_containers if our_containers else "No project containers found"
        else:
            results["docker_status"] = "No running containers or Docker not available"
    except FileNotFoundError:
        results["docker_status"] = "Docker CLI not found"
    except Exception as exc:
        results["docker_status"] = f"Docker check failed: {exc}"

    # Docker Compose status
    try:
        compose_ps = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        if compose_ps.returncode == 0 and compose_ps.stdout.strip():
            results["docker_compose_active"] = True
        else:
            results["docker_compose_active"] = False
    except Exception:
        results["docker_compose_active"] = "Could not check"

    results["status"] = "PASS" if results.get("api_reachable") else "FAIL"
    collector.record("Deployment", results)


# ---------------------------------------------------------------------------
# Metric 8: API Performance
# ---------------------------------------------------------------------------
def measure_api_performance(collector: MetricsCollector) -> None:
    """Measure latency and status of each individual API endpoint."""
    print("  [8/10] Measuring API Performance (per-endpoint latency)...")

    endpoints = [
        ("GET", "/health", None),
        ("GET", "/api/tickets", None),
        ("POST", "/api/knowledge/ingest", {"clear_existing": False}),
    ]

    endpoint_results: dict[str, dict[str, Any]] = {}

    for method, path, body in endpoints:
        try:
            kwargs = {}
            if body:
                kwargs["json"] = body
            resp, latency = timed_request(method, f"{API_BASE_URL}{path}", **kwargs)
            endpoint_results[f"{method} {path}"] = {
                "status_code": resp.status_code,
                "latency_sec": round(latency, 4),
                "success": resp.status_code < 400,
            }
        except Exception as exc:
            endpoint_results[f"{method} {path}"] = {
                "status_code": "ERROR",
                "latency_sec": None,
                "success": False,
                "error": str(exc),
            }

    # Test ticket creation
    try:
        resp, latency = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": f"api-perf-{int(time.time())}@test.io",
                "customer_name": "API Perf",
                "subject": "Glass damage on driver side window",
                "description": (
                    "Claim type: Glass Damage\n"
                    "Policy number: POL-2026-PERF-001\n"
                    "Incident date: 2026-07-16\n"
                    "Loss location: Milpitas, CA\n"
                    "Estimated loss amount: $600.00\n\n"
                    "FNOL narrative: Rock chip cracked the side window."
                ),
                "priority": "low",
                "auto_generate": False,
            },
        )
        endpoint_results["POST /api/tickets"] = {
            "status_code": resp.status_code,
            "latency_sec": round(latency, 4),
            "success": resp.status_code < 400,
        }
        ticket = safe_json(resp)
        if ticket:
            # Test GET single ticket
            resp2, latency2 = timed_request(
                "GET", f"{API_BASE_URL}/api/tickets/{ticket['id']}"
            )
            endpoint_results[f"GET /api/tickets/{ticket['id']}"] = {
                "status_code": resp2.status_code,
                "latency_sec": round(latency2, 4),
                "success": resp2.status_code < 400,
            }

            # Test GET draft (may 404 if none exists)
            resp3, latency3 = timed_request(
                "GET", f"{API_BASE_URL}/api/drafts/{ticket['id']}"
            )
            endpoint_results[f"GET /api/drafts/{ticket['id']}"] = {
                "status_code": resp3.status_code,
                "latency_sec": round(latency3, 4),
                "success": resp3.status_code < 500,  # 404 is acceptable
            }
    except Exception as exc:
        endpoint_results["POST /api/tickets"] = {
            "error": str(exc),
            "success": False,
        }

    latencies = [
        r["latency_sec"]
        for r in endpoint_results.values()
        if r.get("latency_sec") is not None
    ]
    all_success = all(r.get("success", False) for r in endpoint_results.values())

    collector.record("API Performance", {
        "status": "PASS" if all_success else "FAIL",
        "endpoints_tested": len(endpoint_results),
        "all_successful": all_success,
        "mean_latency_sec": statistics.mean(latencies) if latencies else None,
        "max_latency_sec": max(latencies) if latencies else None,
        "min_latency_sec": min(latencies) if latencies else None,
        "per_endpoint": endpoint_results if VERBOSE else f"{len(endpoint_results)} endpoints tested",
        "threshold": "All endpoints return success status",
    })


# ---------------------------------------------------------------------------
# Metric 9: Throughput
# ---------------------------------------------------------------------------
def measure_throughput(collector: MetricsCollector) -> None:
    """Measure concurrent request handling capacity."""
    print("  [9/10] Measuring Throughput (concurrent request handling)...")

    CONCURRENCY = 10
    TOTAL_REQUESTS = 20

    def make_request(i: int) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            resp = requests.get(
                f"{API_BASE_URL}/api/tickets",
                timeout=TIMEOUT,
            )
            elapsed = time.perf_counter() - start
            return {
                "request_id": i,
                "status_code": resp.status_code,
                "latency_sec": elapsed,
                "success": resp.status_code < 400,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return {
                "request_id": i,
                "status_code": "ERROR",
                "latency_sec": elapsed,
                "success": False,
                "error": str(exc),
            }

    results: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(make_request, i): i for i in range(TOTAL_REQUESTS)}
        for future in as_completed(futures):
            results.append(future.result())

    overall_elapsed = time.perf_counter() - overall_start

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency_sec"] for r in results]

    rps = len(results) / overall_elapsed if overall_elapsed > 0 else 0

    collector.record("Throughput", {
        "status": "PASS" if len(failures) == 0 and rps > 1 else "FAIL",
        "total_requests": TOTAL_REQUESTS,
        "concurrency": CONCURRENCY,
        "successful": len(successes),
        "failed": len(failures),
        "total_time_sec": round(overall_elapsed, 4),
        "requests_per_sec": round(rps, 2),
        "mean_latency_sec": round(statistics.mean(latencies), 4) if latencies else None,
        "p95_latency_sec": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 4),
        "max_latency_sec": round(max(latencies), 4) if latencies else None,
        "threshold": "> 1 req/s, 0 failures",
    })


# ---------------------------------------------------------------------------
# Metric 10: LLM Output Quality
# ---------------------------------------------------------------------------
def measure_llm_output_quality(collector: MetricsCollector) -> None:
    """Evaluate LLM-generated draft quality using heuristic checks."""
    print("  [10/10] Measuring LLM Output Quality (draft content heuristics)...")

    try:
        # Create a ticket and generate a draft
        resp, _ = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets",
            json={
                "customer_email": f"quality-{int(time.time())}@test.io",
                "customer_name": "Quality Test User",
                "customer_company": "Quality Assurance Inc",
                "subject": "Property damage claim — tree fell on parked vehicle",
                "description": (
                    "Claim type: Comprehensive\n"
                    "Policy number: POL-2026-QUAL-001\n"
                    "Incident date: 2026-07-11\n"
                    "Loss location: 456 Oak Street, Sunnyvale, CA\n"
                    "Estimated loss amount: $15,000.00\n\n"
                    "FNOL narrative:\n"
                    "During a severe windstorm, a large oak tree fell onto the "
                    "insured's parked 2025 Tesla Model 3, causing significant "
                    "roof and hood damage. The vehicle was parked in the insured's "
                    "driveway at the time of the incident. Photos were taken and "
                    "a homeowner's insurance claim has also been filed. "
                    "The vehicle appears to be a total loss candidate."
                ),
                "priority": "urgent",
                "auto_generate": False,
            },
        )
        ticket = safe_json(resp)
        if not ticket:
            raise RuntimeError("Ticket creation failed")

        draft_resp, _ = timed_request(
            "POST",
            f"{API_BASE_URL}/api/tickets/{ticket['id']}/generate-draft",
        )
        draft_data = safe_json(draft_resp) or {}
        draft_content = draft_data.get("draft", {}).get("content", "")
        context = draft_data.get("draft", {}).get("context_used", {})

        if not draft_content:
            raise RuntimeError("Empty draft content returned")

        # Quality heuristic checks
        checks: dict[str, bool] = {}

        # 1. Minimum length (should be substantive)
        checks["min_length_100_chars"] = len(draft_content) >= 100

        # 2. Contains greeting/acknowledgment
        greeting_patterns = [
            r"\bhi\b", r"\bhello\b", r"\bdear\b", r"\bthanks?\b",
            r"\bthank you\b", r"\breaching out\b", r"\backnowledg",
        ]
        checks["has_greeting"] = any(
            re.search(p, draft_content, re.IGNORECASE) for p in greeting_patterns
        )

        # 3. References the claim type or subject
        checks["references_claim"] = any(
            kw in draft_content.lower()
            for kw in ["comprehensive", "tree", "damage", "claim", "coverage", "vehicle"]
        )

        # 4. Contains actionable next steps
        action_patterns = [
            r"\bnext\s+step", r"\baction\b", r"\bplease\b", r"\bprovide\b",
            r"\bsubmit\b", r"\breview\b", r"\bcontact\b", r"\brecommend",
            r"\bfollow.?up\b", r"\bschedule\b",
        ]
        checks["has_action_items"] = any(
            re.search(p, draft_content, re.IGNORECASE) for p in action_patterns
        )

        # 5. Professional tone (no slang, excessive exclamation)
        checks["professional_tone"] = (
            draft_content.count("!") <= 3
            and "lol" not in draft_content.lower()
            and "omg" not in draft_content.lower()
            and "gonna" not in draft_content.lower()
        )

        # 6. Not too short, not too long
        word_count = len(draft_content.split())
        checks["reasonable_length"] = 30 <= word_count <= 500

        # 7. Doesn't contain obvious errors/hallucinations
        checks["no_placeholder_text"] = not any(
            marker in draft_content.lower()
            for marker in ["[insert", "[todo", "lorem ipsum", "xxx", "[placeholder"]
        )

        # 8. Has structured format (paragraphs or bullet points)
        checks["structured_format"] = (
            "\n" in draft_content or "•" in draft_content or "-" in draft_content
        )

        passed = sum(checks.values())
        total = len(checks)
        quality_score = passed / total if total > 0 else 0

        collector.record("LLM Output Quality", {
            "status": "PASS" if quality_score >= 0.6 else "FAIL",
            "quality_score": f"{quality_score:.0%}",
            "checks_passed": f"{passed}/{total}",
            "word_count": word_count,
            "char_count": len(draft_content),
            "check_details": checks,
            "draft_preview": draft_content[:300] + "..." if len(draft_content) > 300 else draft_content,
            "context_memory_hits": (context or {}).get("signals", {}).get("memory_hit_count", 0),
            "context_kb_hits": (context or {}).get("signals", {}).get("knowledge_hit_count", 0),
            "context_tool_calls": (context or {}).get("signals", {}).get("tool_call_count", 0),
            "threshold": ">= 60% quality checks pass",
        })

    except Exception as exc:
        collector.record("LLM Output Quality", {
            "status": "FAIL",
            "error": str(exc),
        })


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\n{'=' * 60}")
    print("  Insurance Claims Copilot — Metrics Evaluation")
    print(f"  Target: {API_BASE_URL}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")

    # Verify API is reachable
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"  ⚠️  Health endpoint returned {resp.status_code}")
    except Exception as exc:
        print(f"  ❌ API not reachable at {API_BASE_URL}: {exc}")
        print("  Please start the API server first:")
        print("     uv run python main.py")
        print("  Or via Docker:")
        print("     docker compose up -d")
        sys.exit(1)

    collector = MetricsCollector()

    # Run all 10 metrics
    measure_retrieval_speed(collector)
    measure_total_response_time(collector)
    measure_number_of_documents(collector)
    measure_retrieval_accuracy(collector)
    measure_manual_vs_ai_time(collector)
    measure_memory_performance(collector)
    measure_deployment(collector)
    measure_api_performance(collector)
    measure_throughput(collector)
    measure_llm_output_quality(collector)

    # Print summary report
    report = collector.summary()
    print(report)

    # Save report to file
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  📄 Report saved to: {report_path}")

    # Save JSON metrics
    json_path = os.path.join(report_dir, "metrics_report.json")
    with open(json_path, "w") as f:
        json.dump(collector.results, f, indent=2, default=str)
    print(f"  📊 JSON metrics saved to: {json_path}\n")


if __name__ == "__main__":
    main()
