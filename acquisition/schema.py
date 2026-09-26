"""Strict, leakage-resistant storage for the E0--E2 acquisition study.

Public and gold records deliberately live in separate directories.  Selector code
must use :class:`PublicStore`; only evaluation code may instantiate GoldStore.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from acquisition import SCHEMA_VERSION

PUBLIC_TABLES = ("images", "contexts", "candidates", "observations")
GOLD_TABLES = (
    "context_metadata", "entities", "facts", "context_fact_labels", "candidate_fact_labels",
    "observation_claim_labels", "action_labels",
)
SPLITS = {"train", "dev", "test", "excluded"}
FACT_TYPES = {"entity", "attribute", "action", "relation"}
CONTEXT_STATUSES = {"entailed", "contradicted", "not_entailed", "unknown"}
TRINARY = {"yes", "no", "unknown"}
VISIBILITY = {"full", "partial", "none", "unknown"}
CLAIM_STATUSES = {"correct", "contradicted", "unknown"}
GOLD_KEY_PARTS = (
    "fact_id", "entity_id", "target", "label", "utility", "supported_fact",
    "covered_fact", "invalid_claim", "gold", "answer",
)


class SchemaError(ValueError):
    """Raised when an E0 bundle violates the data contract."""


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SchemaError(f"{path}:{line_no}: record must be an object")
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def _clean(row):
    return {key: value for key, value in row.items() if key != "_source_line"}


def _keys(row, required, optional=()):
    actual = set(row) - {"_source_line"}
    missing = set(required) - actual
    extra = actual - set(required) - set(optional)
    if missing or extra:
        raise SchemaError(
            f"line {row.get('_source_line', '?')}: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise SchemaError(f"invalid {name}: {value!r}")


def _opaque_id(value, prefix, name):
    if not isinstance(value, str) or not re.fullmatch(prefix + r"_[0-9a-f]{16,64}", value):
        raise SchemaError(f"{name} must be an opaque hashed identifier")


def _enum(value, allowed, name):
    if value not in allowed:
        raise SchemaError(f"invalid {name}: {value!r}; expected one of {sorted(allowed)}")


def _box(box, width, height):
    if not (isinstance(box, list) and len(box) == 4 and
            all(isinstance(x, int) and not isinstance(x, bool) for x in box)):
        raise SchemaError(f"box must be four integers: {box!r}")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SchemaError(f"invalid half-open box {box!r} for {width}x{height}")


def _no_gold_keys(value, path="record"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(part in lowered for part in GOLD_KEY_PARTS):
                raise SchemaError(f"gold-like public key at {path}.{key}")
            _no_gold_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_gold_keys(child, f"{path}[{index}]")


def canonical_fingerprint(tables):
    """Order-independent content fingerprint for parsed table dictionaries."""
    canonical = {}
    for name, rows in sorted(tables.items()):
        canonical[name] = sorted(
            (_clean(row) for row in rows),
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        )
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def observation_cache_key(image_sha256, candidate, observer_revision, prompt_hash):
    payload = {
        "image_sha256": image_sha256,
        "kind": candidate["kind"],
        "boxes": candidate["boxes"],
        "observer_revision": observer_revision,
        "prompt_hash": prompt_hash,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class PublicStore:
    """The only store that selection methods are allowed to receive."""

    def __init__(self, root):
        self.root = Path(root)
        self.tables = {name: read_jsonl(self.root / "public" / f"{name}.jsonl")
                       for name in PUBLIC_TABLES}
        for rows in self.tables.values():
            for row in rows:
                _no_gold_keys(_clean(row))
        self.images = self._unique_index(self.tables["images"], "image_id", "images")
        self.contexts = self._unique_index(self.tables["contexts"], "context_id", "contexts")
        self.candidates = self._unique_index(self.tables["candidates"], "candidate_id", "candidates")
        self.observations = self._unique_index(self.tables["observations"], "observation_id",
                                               "observations")

    @staticmethod
    def _unique_index(rows, field, table):
        result = {}
        for row in rows:
            key = row.get(field)
            if key in result:
                raise SchemaError(f"{table}: duplicate {field}: {key}")
            result[key] = _clean(row)
        return result

    def selector_state(self, context_id, history_candidate_ids=()):
        """Return model-visible state; unexecuted observations are never exposed."""
        context = self.contexts[context_id]
        image = self.images[context["image_id"]]
        candidates = sorted(
            (row for row in self.candidates.values() if row["image_id"] == image["image_id"]),
            key=lambda row: row["candidate_id"],
        )
        if len(history_candidate_ids) != len(set(history_candidate_ids)):
            raise SchemaError("history repeats an already executed candidate")
        history_ids = set(history_candidate_ids)
        known_candidates = {row["candidate_id"] for row in candidates}
        if not history_ids <= known_candidates:
            raise SchemaError("history contains a candidate from another image")
        history = sorted(
            (row for row in self.observations.values()
             if row["image_id"] == image["image_id"] and row["candidate_id"] in history_ids),
            key=lambda row: row["candidate_id"],
        )
        state = {
            "schema_version": SCHEMA_VERSION,
            "image": {key: image[key] for key in ("image_id", "image_path", "width", "height")},
            "context": {key: context[key] for key in ("context_id", "initial_caption")},
            "candidates": candidates,
            "history": history,
        }
        _no_gold_keys(state)
        return state


class GoldStore:
    """Evaluation-only hidden labels."""

    def __init__(self, root):
        self.root = Path(root)
        self.tables = {name: read_jsonl(self.root / "gold" / f"{name}.jsonl")
                       for name in GOLD_TABLES}


def _validate_bundle_impl(root):
    root = Path(root)
    errors, warnings = [], []
    try:
        public = PublicStore(root)
        gold = GoldStore(root)
        provenance = json.loads((root / "provenance.json").read_text())
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        return {"status": "fail", "errors": [str(exc)], "warnings": []}

    def check_unique(rows, field, table):
        counts = Counter(row.get(field) for row in rows)
        duplicates = sorted(str(key) for key, count in counts.items() if count > 1)
        if duplicates:
            errors.append(f"{table}: duplicate {field}: {duplicates[:5]}")

    for name, rows in public.tables.items():
        check_unique(rows, name[:-1] + "_id", name)
    for name in ("entities", "facts", "observation_claim_labels"):
        singular = {"entities": "entity_id", "facts": "fact_id",
                    "observation_claim_labels": "claim_id"}[name]
        check_unique(gold.tables[name], singular, name)

    images = public.images
    for row in public.tables["images"]:
        try:
            _keys(row, ("image_id", "image_path", "image_sha256", "width", "height", "source",
                        "source_image_id", "duplicate_group_id", "split"))
            _id(row["image_id"], "image_id")
            _opaque_id(row["image_id"], "img", "image_id")
            _enum(row["split"], SPLITS, "split")
            if not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]):
                raise SchemaError("image_sha256 must be lowercase SHA-256")
            if not (isinstance(row["width"], int) and row["width"] > 0 and
                    isinstance(row["height"], int) and row["height"] > 0):
                raise SchemaError("image dimensions must be positive integers")
            path = root / row["image_path"]
            if not path.is_file():
                errors.append(f"missing image file: {row['image_path']}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
                errors.append(f"image hash mismatch: {row['image_id']}")
        except SchemaError as exc:
            errors.append(f"images line {row.get('_source_line')}: {exc}")

    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for image in images.values():
        group_splits[image["duplicate_group_id"]].add(image["split"])
        hash_splits[image["image_sha256"]].add(image["split"])
        source_splits[(image["source"], image["source_image_id"])].add(image["split"])
    for label, mapping in (("duplicate group", group_splits), ("image hash", hash_splits),
                           ("source image", source_splits)):
        for key, splits in mapping.items():
            active = splits - {"excluded"}
            if len(active) > 1:
                errors.append(f"{label} {key!r} crosses splits: {sorted(active)}")

    contexts = public.contexts
    for row in public.tables["contexts"]:
        try:
            _keys(row, ("context_id", "image_id", "initial_caption", "source_kind",
                        "provenance_id"))
            _id(row["context_id"], "context_id")
            _opaque_id(row["context_id"], "ctx", "context_id")
            if row["image_id"] not in images:
                raise SchemaError(f"unknown image_id {row['image_id']}")
            if not isinstance(row["initial_caption"], str) or not row["initial_caption"].strip():
                raise SchemaError("initial_caption must be nonempty")
        except SchemaError as exc:
            errors.append(f"contexts line {row.get('_source_line')}: {exc}")

    context_metadata = {}
    for row in gold.tables["context_metadata"]:
        try:
            _keys(row, ("context_id", "context_kind", "context_group_id",
                        "reference_context_id"))
            if row["context_id"] in context_metadata:
                raise SchemaError(f"duplicate context_id {row['context_id']}")
            if row["context_id"] not in contexts:
                raise SchemaError("unknown context_id")
            _enum(row["context_kind"], {"sparse", "enriched", "paraphrase", "saturated"},
                  "context_kind")
            reference = row["reference_context_id"]
            if reference is not None and reference == row["context_id"]:
                raise SchemaError("context cannot reference itself")
            context_metadata[row["context_id"]] = _clean(row)
        except SchemaError as exc:
            errors.append(f"context_metadata line {row.get('_source_line')}: {exc}")
    if set(context_metadata) != set(contexts):
        errors.append("context_metadata must cover exactly every context")
    for context_id, metadata in context_metadata.items():
        reference = metadata["reference_context_id"]
        if reference is not None and (reference not in contexts or
                                      contexts[reference]["image_id"] != contexts[context_id]["image_id"]):
            errors.append(f"invalid cross-image/missing reference_context_id: {context_id}")

    candidates = public.candidates
    for row in public.tables["candidates"]:
        try:
            _keys(row, ("candidate_id", "image_id", "kind", "boxes", "proposal_generator_revision",
                        "pixel_cost"), ("proposal_score",))
            image = images.get(row["image_id"])
            _opaque_id(row["candidate_id"], "cand", "candidate_id")
            if image is None:
                raise SchemaError(f"unknown image_id {row['image_id']}")
            _enum(row["kind"], {"bbox", "region_set", "stop"}, "candidate kind")
            if not isinstance(row["pixel_cost"], int) or isinstance(row["pixel_cost"], bool) or row["pixel_cost"] < 0:
                raise SchemaError("pixel_cost must be a nonnegative integer")
            if row["kind"] == "stop" and (row["boxes"] or row["pixel_cost"] != 0):
                raise SchemaError("STOP must have no boxes and zero pixel_cost")
            if row["kind"] != "stop" and not row["boxes"]:
                raise SchemaError("non-STOP candidate needs at least one box")
            for box in row["boxes"]:
                _box(box, image["width"], image["height"])
            computed = sum((b[2] - b[0]) * (b[3] - b[1]) for b in row["boxes"])
            if row["pixel_cost"] != computed:
                raise SchemaError(f"pixel_cost {row['pixel_cost']} != box sum {computed}")
        except SchemaError as exc:
            errors.append(f"candidates line {row.get('_source_line')}: {exc}")

    observations = public.observations
    for row in public.tables["observations"]:
        try:
            _keys(row, ("observation_id", "image_id", "candidate_id", "observer_revision",
                        "prompt_hash", "cache_key", "observed_text", "output_tokens", "latency_ms",
                        "status"))
            candidate = candidates.get(row["candidate_id"])
            _opaque_id(row["observation_id"], "obs", "observation_id")
            if candidate is None or candidate["image_id"] != row["image_id"]:
                raise SchemaError("observation candidate/image mismatch")
            _enum(row["status"], {"ok", "failed", "stop"}, "observation status")
            if not isinstance(row["observed_text"], str):
                raise SchemaError("observed_text must be a string")
            for field in ("output_tokens", "latency_ms"):
                if (not isinstance(row[field], int) or isinstance(row[field], bool) or
                        row[field] < 0):
                    raise SchemaError(f"{field} must be a nonnegative integer")
            if not re.fullmatch(r"[0-9a-f]{64}", row["prompt_hash"]):
                raise SchemaError("prompt_hash must be lowercase SHA-256")
            if candidate["kind"] == "stop":
                if row["status"] != "stop" or row["observed_text"] or row["output_tokens"] != 0:
                    raise SchemaError("STOP observation must have stop status, empty text, zero tokens")
            elif row["status"] == "stop":
                raise SchemaError("non-STOP observation cannot have stop status")
            image = images[row["image_id"]]
            expected = observation_cache_key(image["image_sha256"], candidate,
                                             row["observer_revision"], row["prompt_hash"])
            if row["cache_key"] != expected:
                raise SchemaError("observation cache_key mismatch")
        except SchemaError as exc:
            errors.append(f"observations line {row.get('_source_line')}: {exc}")
    observation_counts = Counter(row.get("candidate_id") for row in public.tables["observations"])
    if set(observation_counts) != set(candidates) or any(value != 1 for value in observation_counts.values()):
        errors.append("observations must cover every candidate exactly once")

    entities = {row["entity_id"]: row for row in gold.tables["entities"]}
    facts = {row["fact_id"]: row for row in gold.tables["facts"]}
    for row in gold.tables["entities"]:
        try:
            _keys(row, ("entity_id", "image_id", "category", "evidence_regions", "annotation_status"),
                  ("aliases",))
            if row["image_id"] not in images:
                raise SchemaError("unknown entity image")
            _enum(row["annotation_status"], {"confirmed", "unknown"}, "annotation_status")
            if not isinstance(row["category"], str) or not row["category"].strip():
                raise SchemaError("entity category must be nonempty")
            for box in row["evidence_regions"]:
                image = images[row["image_id"]]
                _box(box, image["width"], image["height"])
        except SchemaError as exc:
            errors.append(f"entities line {row.get('_source_line')}: {exc}")
    for row in gold.tables["facts"]:
        try:
            _keys(row, ("fact_id", "image_id", "fact_type", "subject_entity_id", "predicate",
                        "object", "polarity", "visual_support"))
            _enum(row["fact_type"], FACT_TYPES, "fact_type")
            _enum(row["visual_support"], TRINARY, "visual_support")
            _enum(row["polarity"], {"positive", "negative", "unknown"}, "polarity")
            if not isinstance(row["predicate"], str) or not row["predicate"].strip():
                raise SchemaError("predicate must be nonempty")
            if row["subject_entity_id"] not in entities:
                raise SchemaError("unknown subject entity")
            if entities[row["subject_entity_id"]]["image_id"] != row["image_id"]:
                raise SchemaError("subject belongs to another image")
            obj = row["object"]
            if set(obj) not in ({"literal"}, {"entity_id"}):
                raise SchemaError("fact object needs exactly literal or entity_id")
            if "entity_id" in obj and obj["entity_id"] not in entities:
                raise SchemaError("unknown object entity")
            if "entity_id" in obj and entities[obj["entity_id"]]["image_id"] != row["image_id"]:
                raise SchemaError("object entity belongs to another image")
            if "literal" in obj and (not isinstance(obj["literal"], str) or not obj["literal"].strip()):
                raise SchemaError("literal object must be nonempty")
            if row["fact_type"] == "relation" and "entity_id" not in obj:
                raise SchemaError("relation fact object must be an entity")
        except SchemaError as exc:
            errors.append(f"facts line {row.get('_source_line')}: {exc}")
    if not facts:
        errors.append("fact inventory must be nonempty")
    fact_images = Counter(row.get("image_id") for row in gold.tables["facts"])
    for image_id, image in images.items():
        if image.get("split") != "excluded" and fact_images[image_id] == 0:
            errors.append(f"active image has zero facts: {image_id}")

    seen_context_fact = set()
    for row in gold.tables["context_fact_labels"]:
        try:
            _keys(row, ("context_id", "fact_id", "status", "mention_links", "annotator_ids",
                        "annotator_votes", "adjudication"))
            key = (row["context_id"], row["fact_id"])
            if key in seen_context_fact:
                raise SchemaError(f"duplicate context/fact {key}")
            seen_context_fact.add(key)
            if row["context_id"] not in contexts or row["fact_id"] not in facts:
                raise SchemaError("dangling context/fact reference")
            if contexts[row["context_id"]]["image_id"] != facts[row["fact_id"]]["image_id"]:
                raise SchemaError("context/fact image mismatch")
            _enum(row["status"], CONTEXT_STATUSES, "context fact status")
            if len(row["annotator_votes"]) != 2:
                raise SchemaError("exactly two independent annotator votes are required")
            if (not isinstance(row["annotator_ids"], list) or len(row["annotator_ids"]) != 2 or
                    len(set(row["annotator_ids"])) != 2 or
                    any(not isinstance(value, str) or not value for value in row["annotator_ids"])):
                raise SchemaError("exactly two unique annotator IDs are required")
            for vote in row["annotator_votes"]:
                _enum(vote, CONTEXT_STATUSES, "annotator vote")
            _enum(row["adjudication"], CONTEXT_STATUSES, "adjudication")
            if row["status"] != row["adjudication"]:
                raise SchemaError("canonical status must equal adjudication")
            for link in row["mention_links"]:
                if set(link) != {"span", "entity_id", "resolution"}:
                    raise SchemaError("mention link has invalid keys")
                if link["entity_id"] is not None and link["entity_id"] not in entities:
                    raise SchemaError("mention link references unknown entity")
                _enum(link["resolution"], {"resolved", "ambiguous", "unknown"},
                      "mention resolution")
                span = link["span"]
                if not (isinstance(span, list) and len(span) == 2 and
                        all(isinstance(x, int) and not isinstance(x, bool) for x in span) and
                        0 <= span[0] < span[1] <= len(contexts[row["context_id"]]["initial_caption"])):
                    raise SchemaError("mention span must be valid [start,end] character offsets")
                if link["resolution"] == "resolved" and link["entity_id"] is None:
                    raise SchemaError("resolved mention requires entity_id")
                if (link["entity_id"] is not None and
                        entities[link["entity_id"]]["image_id"] != contexts[row["context_id"]]["image_id"]):
                    raise SchemaError("mention entity belongs to another image")
        except SchemaError as exc:
            errors.append(f"context_fact_labels line {row.get('_source_line')}: {exc}")
    expected_context_fact = {(context_id, fact_id) for context_id, context in contexts.items()
                             for fact_id, fact in facts.items()
                             if fact["image_id"] == context["image_id"]}
    if seen_context_fact != expected_context_fact:
        errors.append("context_fact_labels must exactly cover each same-image context/fact pair")

    seen_candidate_fact = set()
    for row in gold.tables["candidate_fact_labels"]:
        try:
            _keys(row, ("candidate_id", "fact_id", "visibility", "sufficient_to_verify",
                        "annotator_ids", "visibility_annotator_votes",
                        "visibility_adjudication", "annotator_votes", "adjudication"))
            key = (row["candidate_id"], row["fact_id"])
            if key in seen_candidate_fact:
                raise SchemaError(f"duplicate candidate/fact {key}")
            seen_candidate_fact.add(key)
            if row["candidate_id"] not in candidates or row["fact_id"] not in facts:
                raise SchemaError("dangling candidate/fact reference")
            if candidates[row["candidate_id"]]["image_id"] != facts[row["fact_id"]]["image_id"]:
                raise SchemaError("candidate/fact image mismatch")
            _enum(row["visibility"], VISIBILITY, "visibility")
            _enum(row["sufficient_to_verify"], TRINARY, "sufficient_to_verify")
            if len(row["visibility_annotator_votes"]) != 2:
                raise SchemaError("exactly two visibility votes are required")
            for vote in row["visibility_annotator_votes"]:
                _enum(vote, VISIBILITY, "visibility annotator vote")
            _enum(row["visibility_adjudication"], VISIBILITY, "visibility adjudication")
            if row["visibility"] != row["visibility_adjudication"]:
                raise SchemaError("canonical visibility must equal adjudication")
            if len(row["annotator_votes"]) != 2:
                raise SchemaError("exactly two independent annotator votes are required")
            if (not isinstance(row["annotator_ids"], list) or len(row["annotator_ids"]) != 2 or
                    len(set(row["annotator_ids"])) != 2):
                raise SchemaError("exactly two unique annotator IDs are required")
            for vote in row["annotator_votes"]:
                _enum(vote, TRINARY, "candidate annotator vote")
            _enum(row["adjudication"], TRINARY, "candidate adjudication")
            if row["sufficient_to_verify"] != row["adjudication"]:
                raise SchemaError("canonical sufficiency must equal adjudication")
            if row["visibility"] == "none" and row["sufficient_to_verify"] != "no":
                raise SchemaError("invisible fact cannot be sufficient")
            if candidates[row["candidate_id"]]["kind"] == "stop" and (
                    row["visibility"] != "none" or row["sufficient_to_verify"] != "no"):
                raise SchemaError("STOP cannot expose or verify facts")
        except SchemaError as exc:
            errors.append(f"candidate_fact_labels line {row.get('_source_line')}: {exc}")
    expected_candidate_fact = {(candidate_id, fact_id) for candidate_id, candidate in candidates.items()
                               for fact_id, fact in facts.items()
                               if fact["image_id"] == candidate["image_id"]}
    if seen_candidate_fact != expected_candidate_fact:
        errors.append("candidate_fact_labels must exactly cover each same-image candidate/fact pair")

    claims = {}
    for row in gold.tables["observation_claim_labels"]:
        try:
            _keys(row, ("claim_id", "observation_id", "mapped_fact_id", "claim_text", "status",
                        "annotator_ids", "annotator_votes", "adjudication"))
            if row["observation_id"] not in observations:
                raise SchemaError("dangling observation reference")
            if row["mapped_fact_id"] is not None and row["mapped_fact_id"] not in facts:
                raise SchemaError("dangling mapped fact")
            observation = observations[row["observation_id"]]
            if observation["status"] != "ok":
                raise SchemaError("claims are allowed only for successful observations")
            if (row["mapped_fact_id"] is not None and
                    facts[row["mapped_fact_id"]]["image_id"] != observation["image_id"]):
                raise SchemaError("claim mapped fact belongs to another image")
            _enum(row["status"], CLAIM_STATUSES, "claim status")
            if len(row["annotator_votes"]) != 2:
                raise SchemaError("exactly two independent annotator votes are required")
            if (not isinstance(row["annotator_ids"], list) or len(row["annotator_ids"]) != 2 or
                    len(set(row["annotator_ids"])) != 2):
                raise SchemaError("exactly two unique annotator IDs are required")
            for vote in row["annotator_votes"]:
                _enum(vote, CLAIM_STATUSES, "claim annotator vote")
            _enum(row["adjudication"], CLAIM_STATUSES, "claim adjudication")
            if row["status"] != row["adjudication"]:
                raise SchemaError("canonical claim status must equal adjudication")
            claims[row["claim_id"]] = row
        except SchemaError as exc:
            errors.append(f"observation_claim_labels line {row.get('_source_line')}: {exc}")

    derived = derive_action_labels(public, gold)
    supplied = {}
    for row in gold.tables["action_labels"]:
        key = (row.get("context_id"), row.get("candidate_id"))
        if key in supplied:
            errors.append(f"action_labels duplicate context/candidate pair: {key}")
        supplied[key] = _clean(row)
    if set(derived) != set(supplied):
        errors.append("action_labels keys do not cover exactly every context/candidate pair")
    else:
        for key, expected in derived.items():
            if supplied[key] != expected:
                errors.append(f"action label is not derivable from atomic labels: {key}")

    if provenance.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"provenance schema_version must equal {SCHEMA_VERSION}")
    if not provenance.get("automatic_candidate_generator_revision"):
        errors.append("missing automatic_candidate_generator_revision")
    revisions = {row["proposal_generator_revision"] for row in candidates.values()}
    if revisions != {provenance.get("automatic_candidate_generator_revision")}:
        errors.append("candidate generator revision disagrees with provenance")
    unresolved = provenance.get("unresolved_cross_split_near_duplicates", [])
    if unresolved:
        errors.append(f"unresolved cross-split near duplicates: {len(unresolved)}")

    counts = {
        "images": len(images), "contexts": len(contexts), "candidates": len(candidates),
        "observations": len(observations), "entities": len(entities), "facts": len(facts),
        "actions": len(supplied),
    }
    return {
        "status": "pass" if not errors else "fail",
        "schema_version": SCHEMA_VERSION,
        "counts": counts,
        "public_fingerprint": canonical_fingerprint(public.tables),
        "gold_fingerprint": canonical_fingerprint(gold.tables),
        "errors": errors,
        "warnings": warnings,
    }


def validate_bundle(root):
    """Return a failure report, rather than crashing, for malformed external data."""
    try:
        return _validate_bundle_impl(root)
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        return {"status": "fail", "errors": [f"malformed bundle structure: {exc}"],
                "warnings": []}


def derive_action_labels(public, gold):
    """Derive state/action outcomes from atomic labels; never trust hand-written utility."""
    context_status = {(row["context_id"], row["fact_id"]): row["status"]
                      for row in gold.tables["context_fact_labels"]}
    sufficient = {(row["candidate_id"], row["fact_id"]): row["sufficient_to_verify"]
                  for row in gold.tables["candidate_fact_labels"]}
    fact_support = {row["fact_id"]: row["visual_support"] for row in gold.tables["facts"]}
    observation_by_candidate = defaultdict(list)
    for observation in public.observations.values():
        observation_by_candidate[observation["candidate_id"]].append(observation)
    claims_by_observation = defaultdict(list)
    for claim in gold.tables["observation_claim_labels"]:
        claims_by_observation[claim["observation_id"]].append(claim)
    actions = {}
    for context in public.contexts.values():
        image_id = context["image_id"]
        image_fact_ids = sorted(fact_id for fact_id, fact in
                                ((row["fact_id"], row) for row in gold.tables["facts"])
                                if fact["image_id"] == image_id)
        for candidate in sorted((row for row in public.candidates.values()
                                 if row["image_id"] == image_id), key=lambda row: row["candidate_id"]):
            new, repeated, contradicted, unknown = set(), set(), set(), set()
            for observation in observation_by_candidate[candidate["candidate_id"]]:
                for claim in claims_by_observation[observation["observation_id"]]:
                    if claim["status"] == "contradicted":
                        contradicted.add(claim["claim_id"])
                    elif claim["status"] == "unknown" or claim["mapped_fact_id"] is None:
                        unknown.add(claim["claim_id"])
                    elif claim["status"] == "correct":
                        fact_id = claim["mapped_fact_id"]
                        status = context_status.get((context["context_id"], fact_id), "unknown")
                        if status == "not_entailed" and fact_support.get(fact_id) == "yes" and \
                                sufficient.get((candidate["candidate_id"], fact_id)) == "yes":
                            new.add(fact_id)
                        elif status == "entailed":
                            repeated.add(fact_id)
                        else:
                            # A mapped claim must be conserved even when context
                            # coverage or visual sufficiency is unresolved.
                            unknown.add(claim["claim_id"])
            actions[(context["context_id"], candidate["candidate_id"])] = {
                "context_id": context["context_id"],
                "candidate_id": candidate["candidate_id"],
                "new_supported_fact_ids": sorted(new),
                "repeated_fact_ids": sorted(repeated),
                "contradicted_claim_ids": sorted(contradicted),
                "unknown_claim_ids": sorted(unknown),
                "eligible_missing_fact_ids": sorted(
                    fact_id for fact_id in image_fact_ids
                    if fact_support.get(fact_id) == "yes" and
                    context_status.get((context["context_id"], fact_id)) == "not_entailed"
                ),
                "pixel_cost": candidate["pixel_cost"],
            }
    return actions
