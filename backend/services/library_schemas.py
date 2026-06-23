"""Pydantic models for the on-disk JSON formats stored in the public HF
registry.

Every record pulled from HF is validated against one of these models before
it touches the local DB. Malformed payloads from a misbehaving publisher,
schema-evolution mismatches, or partial downloads all surface here as a
ValidationError instead of corrupting local state.

Forward-compatibility policy: every model uses `extra="ignore"`. Future
fields land in v2 records, get silently dropped by v1 clients, and the
v1 client keeps working. New required fields would force a schema_version
bump and a coordinated client release.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class Manifest(BaseModel):
    """Index of every record currently in the registry. Maps public_id to
    the record's published_at timestamp.

    `categories_hash` is the sha256 of categories.json content (stable JSON
    encoding). When categories change, this hash changes, so the manifest's
    own content hash changes — meaning the cheap-probe in sync_library still
    detects categories-only updates without any extra plumbing.

    `rule_names` and `ce_names` are reverse lookups: name -> public_id. They
    let publishers detect "this name is already taken" without scanning every
    record file. Required to prevent same-name collisions when AI pipelines
    on different machines independently generate the same name. Old code that
    doesn't read these fields still works (default empty).

    `ce_calibration` carries the per-CE calibration conversations the upstream
    GAVEL research project ships alongside each CE, mapping the CE's public_id
    to the calibration record's published_at. Old clients that don't read it
    get an empty default and keep working.

    Two earlier sections were removed: `rule_evaluation` (rolled back — eval
    sets are per-experiment) and `rule_calibration` (the old per-rule
    usecase-level calibration upload, replaced by `rule_datasets` below). Old
    manifests carrying either are silently ignored thanks to extra="ignore".
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    rules: Dict[str, str] = {}
    ces: Dict[str, str] = {}
    rule_names: Dict[str, str] = {}
    ce_names: Dict[str, str] = {}
    categories_hash: Optional[str] = None
    # neutral (schema v11): the global neutral corpus, split into its two
    # pseudo use-cases. Maps category ("conversational" / "instructive") -> the
    # sha256 of that category's neutral/<category>/conversations.json. A changed
    # hash tells sync to re-pull just that category. Old clients ignore it.
    neutral: Dict[str, str] = {}
    ce_calibration: Dict[str, str] = {}
    # rule_datasets (schema v9): maps a rule's public_id -> published_at for
    # its DEFAULT test/calibration set. Each published rule has three dataset
    # files (positive / negative / positive_calibration) sharing this stamp;
    # one manifest entry per rule is enough for sync to know they exist and
    # whether the local copy is stale. Old clients ignore this field.
    rule_datasets: Dict[str, str] = {}
    # rule_sets (schema v19): a published rule set's public_id -> published_at.
    # rule_set_names is the reverse name -> public_id index used for same-name
    # collision detection at publish time (mirrors rule_names / ce_names). A
    # rule set is a thin pointer-collection of already-published rule public_ids;
    # its member rules live under public_rules/ and are pulled lazily. Old
    # clients ignore both fields (default empty).
    rule_sets: Dict[str, str] = {}
    rule_set_names: Dict[str, str] = {}


class Category(BaseModel):
    """One entry in the shared category vocabulary. Identified by name —
    categories don't fork or version like rules and CEs because they're
    a controlled vocabulary, not user-authored content."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""


class CategoriesFile(BaseModel):
    """Wrapper for the categories.json root file."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    categories: List[Category] = []


class NeutralCorpusFile(BaseModel):
    """Wrapper for a neutral/<category>/conversations.json file. Each entry in
    `conversations` is a message list ([{role, content}, ...])."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    category: str = ""
    conversations: List[List[Dict[str, str]]] = []


class CERecord(BaseModel):
    """A published cognitive element. The matching excitation_<public_id>.json
    file is guaranteed to exist in the registry alongside this one."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    public_id: str
    name: str
    definition: str = ""
    category: str = "CONTEXT"
    categories: List[str] = []
    examples: List[Any] = []
    published_at: Optional[str] = None
    # Phase 1 creator attribution. Missing on pre-feature artifacts;
    # the sync upsert defaults to the configured seed-team username
    # when this is None so Browse pages don't show "anonymous".
    created_by_username: Optional[str] = None


class RuleRecord(BaseModel):
    """A published rule. ce_dependencies is the list of public_ids of CEs
    referenced by this rule; the role lists (necessary / fallback /
    sufficient) hold CE *names* and are wired to the local DB via name
    lookup at insert time."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    public_id: str
    name: str
    predicate: str = ""
    necessary: List[str] = []
    fallback: List[List[str]] = []
    sufficient: List[str] = []
    categories: List[str] = []
    definition: str = ""
    ce_dependencies: List[str] = []
    published_at: Optional[str] = None
    # Phase 1 creator attribution. Same fallback semantics as CERecord.
    created_by_username: Optional[str] = None


class RuleSetRecord(BaseModel):
    """A published rule set (schema v19): a named, attributed, model-agnostic
    collection of already-published rules, stored at
    public_rule_sets/{public_id}.json.

    `member_rules` is the ORDERED list of member rule public_ids — a thin
    pointer collection, exactly as RuleRecord.ce_dependencies references CEs by
    public_id. The member rules' definitions, CEs, and default datasets live in
    public_rules/ etc. and are pulled lazily; they are NOT embedded here. The
    target model, training data, thresholds, and metrics are NEVER serialized —
    only the model-agnostic rule selection is shared. `categories` holds
    category NAMES (not local int ids), like the other records."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    public_id: str
    name: str
    description: str = ""
    categories: List[str] = []
    member_rules: List[str] = []
    published_at: Optional[str] = None
    # Phase 1 creator attribution. Same fallback semantics as CERecord/RuleRecord.
    created_by_username: Optional[str] = None


class ExcitationRecord(BaseModel):
    """The training data attached to a published CE. samples is preserved
    as the raw conversation list — the guardrail engine consumes this
    shape directly."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    ce_public_id: str
    samples: List[Any] = []
    sample_count: int = 0
    published_at: Optional[str] = None


class CECalibrationRecord(BaseModel):
    """CE-level calibration conversations attached to a published CE.
    Same shape as ExcitationRecord but used by the calibration pipeline
    (Youden-J threshold sweep) rather than training. The seed library
    ships these so a fresh client can run `evaluate` end-to-end without
    generating calibration data from scratch."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    ce_public_id: str
    samples: List[Any] = []
    sample_count: int = 0
    published_at: Optional[str] = None


class RuleDatasetRecord(BaseModel):
    """A rule's DEFAULT test/calibration dialogues (schema v9). Three of
    these are published per public rule — positive, negative, and
    positive_calibration — under public_rule_datasets/{rule_pid}_{type}.json.

    Carries DIALOGUES + the generation config only. Never thresholds or
    metrics: those are computed per-guardrail against the adopter's own
    trained model, so they can't be shared. The scenario lives inside
    `config.scenario_instructions`."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    rule_public_id: str
    dataset_type: str  # 'positive' | 'negative' | 'positive_calibration'
    config: Dict[str, Any] = {}
    conversations: List[Any] = []
    conversation_count: int = 0
    published_at: Optional[str] = None


