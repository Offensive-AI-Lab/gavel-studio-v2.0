from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sql_scripts.user_scripts import sync_user_to_local
from services import central_server
from services.central_server import CentralServerError
from utils.text_safety import clean_text, validate_username

_bearer = HTTPBearer(auto_error=False)


def _get_bearer_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
	"""Returns the raw bearer token for forwarding to the central server."""
	if not creds:
		raise HTTPException(status_code=401, detail="Missing bearer token")
	return creds.credentials

router = APIRouter()


def _raise_for_central(err: CentralServerError):
    """Translate central-server errors into FastAPI HTTPExceptions."""
    raise HTTPException(status_code=err.status_code, detail=str(err))


def _mirror_team_users() -> None:
    """Pull every is_team user from the central server and upsert them
    into the local users table. Called on register + login so any
    fresh local DB ends up with rows for the seed-library author(s) —
    without these rows, rules/CEs they authored can't JOIN to a display
    name and the @<team> profile shows an empty contributions list.

    Best-effort: a central outage must not block the user's login, so
    every failure is swallowed and logged."""
    try:
        users = central_server.get_team_users() or []
        for u in users:
            try:
                sync_user_to_local(u)
            except Exception as inner:
                print(f"[auth] Failed to mirror team user {u.get('username')!r}: {inner}")
    except Exception as e:
        print(f"[auth] Could not fetch team users from central: {e}")

# Request models
class RegisterRequest(BaseModel):
	username: str = Field(..., max_length=30)
	email: str = Field(..., max_length=254)
	password: str = Field(..., min_length=8, max_length=128)

	@field_validator("username", mode="before")
	@classmethod
	def _validate_username(cls, value):
		return validate_username(value)

	@field_validator("email", mode="before")
	@classmethod
	def _normalize_email(cls, value):
		return clean_text(value, field_name="email", max_length=254)

class LoginRequest(BaseModel):
	email: str = Field(..., max_length=254)
	password: str = Field(..., min_length=8, max_length=128)

	@field_validator("email", mode="before")
	@classmethod
	def _normalize_login_email(cls, value):
		return clean_text(value, field_name="email", max_length=254)

# Response models
class RegisterResponse(BaseModel):
	status: str
	user_id: int
	username: str
	email: str

class LoginResponse(BaseModel):
	status: str
	token: str
	user_id: int
	username: str
	email: str
	# First-login tutorial flag. The Workspace page auto-opens the
	# onboarding modal when this is False on mount.
	tutorial_seen: bool = False

@router.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
	"""Forwards registration to the central server. Username/email
	uniqueness is enforced there; password hashing happens there too.

	On success, the new user is mirrored to the local DB so FK
	constraints in target_models, etc. resolve."""
	try:
		user = central_server.register(req.username, req.email, req.password)
	except CentralServerError as err:
		_raise_for_central(err)

	# Mirror the new user into the local DB
	sync_user_to_local(user)
	_mirror_team_users()

	return RegisterResponse(
		status="ok",
		user_id=user["user_id"],
		username=user["username"],
		email=user["email"],
	)

@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
	"""Authenticates against the central server, which signs the token.
	The local backend does NOT hold the signing secret — its
	`get_current_user` dependency validates tokens by calling the central
	server's /auth/verify, so no JWT secret needs to live in the local env."""
	try:
		result = central_server.login(req.email, req.password)
	except CentralServerError as err:
		_raise_for_central(err)

	# Mirror user data into local DB
	sync_user_to_local(result)
	_mirror_team_users()

	return LoginResponse(
		status="ok",
		token=result["token"],
		user_id=result["user_id"],
		username=result["username"],
		email=result["email"],
		tutorial_seen=bool(result.get("tutorial_seen") or False),
	)


@router.put("/tutorial-seen")
def mark_tutorial_seen(token: str = Depends(_get_bearer_token)):
	"""Flip the per-user tutorial_seen flag to TRUE.

	Called when the user finishes or skips the first-login onboarding
	modal. Idempotent — re-calls are no-ops because the SET is
	unconditional. The frontend caches the localStorage user object
	and updates its tutorial_seen field after a successful response so
	a re-mount of /workspace doesn't re-trigger the modal."""
	try:
		central_server.mark_tutorial_seen(token)
	except CentralServerError as err:
		_raise_for_central(err)
	return {"status": "ok"}


@router.get("/me")
def get_me(token: str = Depends(_get_bearer_token)):
	"""Returns the authenticated user's profile from the central server."""
	try:
		user = central_server.get_me(token)
	except CentralServerError as err:
		_raise_for_central(err)
	# Refresh local mirror with the latest profile data
	sync_user_to_local(user)
	return user


# ---------------------------------------------------------------------------
# Phase 2: public profile + edit-own-profile endpoints.
# ---------------------------------------------------------------------------


class ProfileResponse(BaseModel):
	"""Public-profile shape. Includes aggregate stats derived from
	user_ratings_summary so the profile page can render counts and
	average rating without a second round-trip.

	Note on email visibility: this endpoint is unauthenticated and
	returns email for every profile, so emails are effectively public.
	For the GAVEL team account that's fine (a published contact
	address). For regular users, surfacing email on a public profile
	creates a scrape/spam target — revisit if/when the project grows
	beyond the original cohort."""
	user_id: int
	username: str
	email: Optional[str] = None
	display_name: Optional[str] = None
	bio: Optional[str] = None
	is_team: bool
	member_since: Optional[str] = None
	contribution_count_rules: int = 0
	contribution_count_ces: int = 0
	total_rating_count: int = 0
	avg_rating_received: Optional[float] = None
	last_published_at: Optional[str] = None


class UpdateMeRequest(BaseModel):
	"""Editable profile fields. Username + email are intentionally NOT
	here — username is permanent per design, and email changes need a
	separate verification flow we haven't built."""
	display_name: Optional[str] = Field(None, max_length=255)
	bio: Optional[str] = Field(None, max_length=2000)


def _local_published_counts(usernames: list) -> dict:
	"""Map username(lowercased) -> {"rules": n, "ces": n} of PUBLISHED items in
	the LOCAL synced library.

	The community count number used to come from the central server's
	publish-time counter, which DRIFTS when items are removed from HF outside
	the app (it is never decremented). The profile contributions *list* already
	reads the local synced DB, so we count from the same place — keeping the
	number consistent with the list and accurate to the current library state.
	"""
	from utils.PostgreSQL import execute_query_dict
	unames = sorted({(u or "").strip().lower() for u in usernames if u and str(u).strip()})
	if not unames:
		return {}
	out = {u: {"rules": 0, "ces": 0} for u in unames}
	rules = execute_query_dict(
		"SELECT LOWER(created_by_username) AS u, COUNT(*) AS n FROM rules "
		"WHERE LOWER(created_by_username) = ANY(%s) AND is_local_draft = FALSE GROUP BY 1",
		(unames,),
	) or []
	ces = execute_query_dict(
		"SELECT LOWER(created_by_username) AS u, COUNT(*) AS n FROM cognitive_elements "
		"WHERE LOWER(created_by_username) = ANY(%s) AND is_local_draft = FALSE GROUP BY 1",
		(unames,),
	) or []
	for r in rules:
		out.setdefault(r["u"], {"rules": 0, "ces": 0})["rules"] = r["n"]
	for c in ces:
		out.setdefault(c["u"], {"rules": 0, "ces": 0})["ces"] = c["n"]
	return out


def _apply_local_counts(item: dict) -> dict:
	"""Override an artist/profile dict's contribution counts with the LOCAL
	synced-library counts (accurate vs the drift-prone central counter)."""
	uname = (item.get("username") or "").strip().lower()
	c = _local_published_counts([uname]).get(uname, {"rules": 0, "ces": 0})
	item["contribution_count_rules"] = c["rules"]
	item["contribution_count_ces"] = c["ces"]
	return item


def _synced_artists(items: list) -> list:
	"""Filter a central-server artist list down to the contributors whose
	published work is actually in THIS backend's locally-synced library, and
	stamp each with its accurate local counts.

	Discovery (search + leaderboard) is sourced from the central server, which
	knows every user who has ever published. But the user's mental model is that
	the community reflects *their* synced library: a new contributor should
	appear only AFTER they run Sync and that person's rule/CE lands locally —
	never as a pre-sync "0 contributions" ghost. So we drop anyone with zero
	local published items here."""
	local = _local_published_counts([it.get("username") for it in items])
	out = []
	for it in items:
		c = local.get((it.get("username") or "").strip().lower(), {"rules": 0, "ces": 0})
		if (c["rules"] + c["ces"]) <= 0:
			continue  # not in the local synced library yet → hide until Sync pulls them
		it["contribution_count_rules"] = c["rules"]
		it["contribution_count_ces"] = c["ces"]
		out.append(it)
	return out


@router.get("/profile/{username}", response_model=ProfileResponse)
def get_profile(username: str):
	"""Public profile lookup. Identity + ratings come from the central server;
	the contribution COUNTS are recomputed from the local synced library so they
	stay accurate even after items were removed from HF."""
	username = (username or "").strip().lower()
	if not username:
		raise HTTPException(status_code=400, detail="username required")
	try:
		row = central_server.get_profile(username)
	except CentralServerError as err:
		_raise_for_central(err)
	return ProfileResponse(**_apply_local_counts(dict(row)))


@router.get("/profile/{username}/contributions")
def get_profile_contributions(
	username: str,
	type: str = "rule",
	page: int = 1,
	page_size: int = 20,
):
	"""Paginated list of a user's published rules or CEs. Drafts are
	excluded — this endpoint reflects the **public** library only."""
	from utils.PostgreSQL import execute_query_dict
	username = (username or "").strip().lower()
	if type not in ("rule", "ce"):
		raise HTTPException(status_code=400, detail="type must be 'rule' or 'ce'")
	if page < 1:
		raise HTTPException(status_code=400, detail="page must be >= 1")
	if page_size < 1 or page_size > 100:
		raise HTTPException(status_code=400, detail="page_size must be 1-100")

	# Existence check goes to central server (source of truth for users).
	try:
		central_server.get_profile(username)
	except CentralServerError as err:
		_raise_for_central(err)

	offset = (page - 1) * page_size
	if type == "rule":
		total = execute_query_dict(
			"SELECT COUNT(*) AS n FROM rules "
			"WHERE created_by_username = %s AND is_local_draft = FALSE",
			(username,),
		)[0]["n"]
		items = execute_query_dict(
			"""
			SELECT rule_id AS id, name, predicate AS content,
				   public_id, published_at, categories
			FROM rules
			WHERE created_by_username = %s AND is_local_draft = FALSE
			ORDER BY published_at DESC NULLS LAST, rule_id DESC
			LIMIT %s OFFSET %s
			""",
			(username, page_size, offset),
		)
	else:
		total = execute_query_dict(
			"SELECT COUNT(*) AS n FROM cognitive_elements "
			"WHERE created_by_username = %s AND is_local_draft = FALSE",
			(username,),
		)[0]["n"]
		items = execute_query_dict(
			"""
			SELECT ce_id AS id, name, definition AS content,
				   public_id, published_at, categories
			FROM cognitive_elements
			WHERE created_by_username = %s AND is_local_draft = FALSE
			ORDER BY published_at DESC NULLS LAST, ce_id DESC
			LIMIT %s OFFSET %s
			""",
			(username, page_size, offset),
		)

	# Render datetimes / lists into JSON-safe primitives.
	for it in (items or []):
		if it.get("published_at"):
			it["published_at"] = it["published_at"].isoformat()
		# `categories` comes back as an array of ints (the join through
		# categories.json runs frontend-side in browse, and the profile
		# page can do the same — keeping this endpoint cheap).
		it["categories"] = it.get("categories") or []

	return {
		"username": username,
		"type": type,
		"page": page,
		"page_size": page_size,
		"total": total,
		"items": items or [],
	}


@router.patch("/me")
def update_me(req: UpdateMeRequest, token: str = Depends(_get_bearer_token)):
	"""Edit your own profile. Forwards to the central server (source of
	truth) and refreshes the local mirror."""
	display_name = clean_text(req.display_name, field_name="display_name", max_length=255) if req.display_name else req.display_name
	bio = clean_text(req.bio, field_name="bio", max_length=2000, allow_newlines=True) if req.bio else req.bio

	try:
		user = central_server.update_me(token, display_name=display_name, bio=bio)
	except CentralServerError as err:
		_raise_for_central(err)

	if user:
		sync_user_to_local(user)
	return user or {}


# ---------------------------------------------------------------------------
# Phase 4: discovery — user search + leaderboard.
#
# Both endpoints enforce the "artist gate": users with zero contributions
# don't appear in either listing. This is the Spotify-like distinction
# between "listeners" (registered but haven't published) and "artists"
# (have published at least one rule or CE). Direct profile lookup by
# URL still works for listeners — only discovery is gated.
# ---------------------------------------------------------------------------


class ArtistSummary(BaseModel):
	"""Compact card shape used by both search and leaderboard. The
	frontend renders a list of these without needing a second call."""
	username: str
	display_name: Optional[str] = None
	bio: Optional[str] = None
	is_team: bool
	contribution_count_rules: int
	contribution_count_ces: int
	total_rating_count: int
	avg_rating_received: Optional[float] = None


class ArtistListResponse(BaseModel):
	page: int
	page_size: int
	total: int
	items: List[ArtistSummary]


def _row_to_artist(row: dict) -> ArtistSummary:
	"""Helper shared by search + leaderboard. Computes avg from the
	summary table's count + sum so callers don't repeat the math."""
	count = row.get("total_rating_count") or 0
	rsum = row.get("total_rating_sum") or 0
	avg = round(rsum / count, 2) if count > 0 else None
	return ArtistSummary(
		username=row["username"],
		display_name=row.get("display_name"),
		bio=row.get("bio"),
		is_team=bool(row.get("is_team")),
		contribution_count_rules=row.get("contribution_count_rules") or 0,
		contribution_count_ces=row.get("contribution_count_ces") or 0,
		total_rating_count=count,
		avg_rating_received=avg,
	)


# How many central candidates we pull before applying the local-sync gate.
# We page LOCALLY after gating (so totals/pages reflect only synced contributors),
# which means we need the full candidate set up front. 100 is the central cap and
# is far more than this single-user platform's realistic contributor count.
_DISCOVERY_FETCH = 100


def _paginate(gated: list, page: int, page_size: int) -> ArtistListResponse:
	"""Slice an already-gated artist list for the requested page. `total` is the
	gated length so the UI's pager reflects only locally-synced contributors."""
	start = (page - 1) * page_size
	page_items = gated[start:start + page_size]
	return ArtistListResponse(
		page=page,
		page_size=page_size,
		total=len(gated),
		items=[ArtistSummary(**it) for it in page_items],
	)


@router.get("/search", response_model=ArtistListResponse)
def search_artists(q: str = "", page: int = 1, page_size: int = 20):
	"""Find users who have published at least one artifact. Candidates come from
	the central server (source of truth for users + ratings summary); we then keep
	only contributors whose work is in the LOCAL synced library (see
	_synced_artists) and page over that gated set."""
	if page < 1 or page_size < 1 or page_size > 100:
		raise HTTPException(status_code=400, detail="invalid pagination")
	try:
		resp = central_server.search_users(q=q.strip() or None, page=1, page_size=_DISCOVERY_FETCH)
	except CentralServerError as err:
		_raise_for_central(err)
	gated = _synced_artists([dict(it) for it in resp.get("items", [])])
	return _paginate(gated, page, page_size)


@router.get("/leaderboard", response_model=ArtistListResponse)
def leaderboard(by: str = "avg_rating", page: int = 1, page_size: int = 20, min_ratings: int = 0):
	"""Top contributors. Candidates come from the central server; we then keep only
	contributors whose work is in the LOCAL synced library and page over that set.

	`min_ratings` is a caller-controlled floor on ratings received (the Community
	"minimum ratings" filter) — forwarded to the central server."""
	if by not in ("avg_rating", "count"):
		raise HTTPException(status_code=400, detail="by must be 'avg_rating' or 'count'")
	if page < 1 or page_size < 1 or page_size > 100:
		raise HTTPException(status_code=400, detail="invalid pagination")
	if min_ratings < 0:
		raise HTTPException(status_code=400, detail="min_ratings must be >= 0")
	sort = "rating" if by == "avg_rating" else "contributions"
	try:
		resp = central_server.leaderboard(sort=sort, page=1, page_size=_DISCOVERY_FETCH, min_ratings=min_ratings)
	except CentralServerError as err:
		_raise_for_central(err)
	gated = _synced_artists([dict(it) for it in resp.get("items", [])])
	return _paginate(gated, page, page_size)
