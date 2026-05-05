# Bia LiteLLM Operator Manual

This document is the human-oriented operating manual for the Bia LiteLLM gateway currently deployed on `gpustack.ing.unibs.it` (`bia001`).

It focuses on the questions that matter in day-to-day operation:

- what is running
- where configuration lives
- how to create users
- how student access should be managed
- how passwords work
- how to give students UI access with limited permissions
- how to create, block, and budget API keys

## 1. Current deployment

The current production entrypoint is:

- API/UI base URL: `https://gpustack.ing.unibs.it`
- OpenAI-compatible API: `https://gpustack.ing.unibs.it/v1`
- LiteLLM UI: `https://gpustack.ing.unibs.it/ui/`

The current deployment has:

- LiteLLM gateway on `bia001`
- PostgreSQL for LiteLLM metadata on `bia001`
- one `llama.cpp` backend per Bia host
- student-facing model pool excluding `bia014` and `bia015`
- experimental unaligned models only on `bia014` and `bia015`

### Current model aliases

These are the model names exposed by the LiteLLM gateway today.

| Alias | Type | Access | Notes |
| --- | --- | --- | --- |
| `qwen35-4b` | chat/completions | standard | general chat/coding |
| `ministral3-3b` | chat/completions | standard | general chat/coding |
| `phi4-mini` | chat/completions | standard | general chat/coding |
| `embeddinggemma` | embeddings | standard | text embeddings |
| `llamaguard3-1b` | chat/completions | security-scoped only | guardrail / safety analysis |
| `granite-guardian-2b` | chat/completions | security-scoped only | guardrail / safety analysis |
| `experimental-heretic-llama32-3b` | chat/completions | instructor/security-scoped only | experimental unaligned model |
| `experimental-abliterated-llama32-3b` | chat/completions | instructor/security-scoped only | experimental unaligned model |

The hostnames `bia012`-`bia015` identify where some restricted models run. They are not the API model names clients should send to LiteLLM.

### Client usage examples

Set your client variables:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://gpustack.ing.unibs.it/v1"
```

Chat completion with `curl`:

```bash
curl -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "model": "qwen35-4b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Give me three ideas for a classroom exercise on prompt engineering."}
    ]
  }'
```

Embeddings with `curl`:

```bash
curl -sS "$OPENAI_BASE_URL/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "model": "embeddinggemma",
    "input": "LiteLLM on the Bia cluster"
  }'
```

Python example with the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="https://gpustack.ing.unibs.it/v1",
)

chat = client.chat.completions.create(
    model="phi4-mini",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarize the benefits of using a LiteLLM gateway."},
    ],
)
print(chat.choices[0].message.content)

embedding = client.embeddings.create(
    model="embeddinggemma",
    input="LiteLLM on the Bia cluster",
)
print(len(embedding.data[0].embedding))
```

Important current fact:

- the gateway `.env` currently contains only:
  - `POSTGRES_PASSWORD`
  - `LITELLM_MASTER_KEY`
  - `LITELLM_SALT_KEY`
- `UI_USERNAME` and `UI_PASSWORD` are not currently configured

That means there is not currently a shared static username/password gate configured in front of the UI.

## 2. Where things live

On `bia001`, as user `litellm`:

- LiteLLM root: `/home/litellm/bia-litellm`
- Compose file: `/home/litellm/bia-litellm/docker-compose.yml`
- LiteLLM config: `/home/litellm/bia-litellm/config.yaml`
- secrets: `/home/litellm/bia-litellm/.env`
- generated virtual keys: `/home/litellm/bia-litellm/generated-keys.json`

TLS files used by the gateway:

- `/etc/ssl/gpustack/cert.pem`
- `/etc/ssl/gpustack/key.pem`

## 3. Two different kinds of access

There are two separate concepts. Do not mix them up.

### A. API access

This is for calling `/v1/chat/completions`, `/v1/embeddings`, and so on.

API access is controlled by LiteLLM virtual keys:

- keys look like `sk-...`
- each key can be limited by:
  - allowed models
  - budget
  - duration
  - RPM/TPM
  - user
  - team

### B. Web UI access

This is for logging into the LiteLLM web interface.

UI access is controlled by LiteLLM internal users and roles, not by normal API keys alone.

For students, this is the important distinction:

- if you want them to call models from code, give them API keys
- if you want them to log into the UI, create LiteLLM internal users and invite them

## 4. How passwords work

There are two possible password systems in LiteLLM.

### A. Global UI username/password

LiteLLM supports a single shared UI gate via:

- `UI_USERNAME`
- `UI_PASSWORD`

This is a platform-wide shared credential. It is not per-student.

Use it only if you want one extra coarse gate in front of `/ui/`.

It is not a good primary mechanism for a class, because everyone would share the same login.

### B. Per-user UI login

LiteLLM also supports invited internal users. This is the correct model for students.

The documented flow is:

1. create an internal user
2. generate an invitation link
3. the user completes onboarding
4. the user logs in via email + password auth

This is the right path if each student should have their own account.

### Password reset / password recovery

I did not find an official LiteLLM doc page that documents a dedicated password-reset endpoint for invited internal users.

Operationally, the safe assumption is:

- the onboarding/invitation flow is the supported way to establish per-user credentials
- if a student loses access, the pragmatic recovery path is to issue a fresh invitation link and have them re-onboard
- if the current LiteLLM UI version exposes a direct reset flow, treat that as convenience, not as the only recovery path

For operations, assume invitation re-issue is the reliable fallback.

## 5. Recommended student model

For your use case, the cleanest setup is:

1. create one LiteLLM team per course or lab cohort
2. put the student-allowed models on that team
3. assign a team budget
4. assign per-student team-member budgets
5. invite students as UI users
6. restrict what the UI shows them

This is better than giving every student unrestricted personal keys.

Reason:

- budget control is cleaner
- model access is cleaner
- student traffic is grouped
- you keep `bia014` and `bia015` out of normal student access

## 6. Access profiles

Use explicit access profiles. Do not rely on informal naming or trust alone.

| Profile | Who | UI role | Team role | Default models | Can manage users/keys? |
| --- | --- | --- | --- | --- | --- |
| `academic-standard` | fellow academics | `internal_user` | team member with key-management permissions | `qwen35-4b`, `ministral3-3b`, `phi4-mini`, `embeddinggemma` | yes, keys inside their team |
| `student-standard` | normal course students | `internal_user_viewer` | team member | `qwen35-4b`, `ministral3-3b`, `phi4-mini`, `embeddinggemma` | no |
| `student-security` | Computer Security students | `internal_user_viewer` or `internal_user` | team member | standard models plus `llamaguard3-1b`, `granite-guardian-2b`, `experimental-heretic-llama32-3b`, `experimental-abliterated-llama32-3b` | usually no |
| `security-instructor` | trusted security instructors | `internal_user` | security-scoped team member with key-management permissions | standard models plus `llamaguard3-1b`, `granite-guardian-2b`, `experimental-heretic-llama32-3b`, `experimental-abliterated-llama32-3b` | yes, keys inside their team |
| `platform-admin` | Bia/LiteLLM operators | `proxy_admin` | global | all configured models | yes, globally |

Important rule:

- do not make fellow academics `proxy_admin` by default
- create a team for each academic or course
- add the academic as a normal member of that team
- grant team-member key-management permissions on academic teams
- restrict the team's `models` list to what they should be able to use

This lets academics manage their own keys without giving them platform-wide control or accidental access to the restricted security and experimental model aliases.

## 7. Model access groups

### Standard models

Use this allowlist for fellow academics and normal courses:

```json
["qwen35-4b", "ministral3-3b", "phi4-mini", "embeddinggemma"]
```

This excludes:

- `llamaguard3-1b`
- `granite-guardian-2b`
- `experimental-heretic-llama32-3b`
- `experimental-abliterated-llama32-3b`

### Computer Security models

Use this allowlist only for Computer Security courses, security instructors, or explicitly approved research work:

```json
[
  "qwen35-4b",
  "ministral3-3b",
  "phi4-mini",
  "embeddinggemma",
  "llamaguard3-1b",
  "granite-guardian-2b",
  "experimental-heretic-llama32-3b",
  "experimental-abliterated-llama32-3b"
]
```

Operational rule:

- `llamaguard3-1b` and `granite-guardian-2b` are guardrail/security-analysis models
- `experimental-heretic-llama32-3b` and `experimental-abliterated-llama32-3b` are experimental unaligned models
- only security-scoped teams should receive those aliases

## 8. Fellow academic workflow

For a fellow academic, create an internal user and a dedicated team. Add the academic as a team member with key-management permissions, not as a global platform admin.

### Create the academic user

```bash
curl -ksS "$PROXY/user/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "user_email": "academic@example.org",
    "user_role": "internal_user"
  }'
```

### Create the academic team

```bash
curl -ksS "$PROXY/team/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "team_alias": "academic-example-lab",
    "models": ["qwen35-4b", "ministral3-3b", "phi4-mini", "embeddinggemma"],
    "team_member_permissions": [
      "/key/generate",
      "/key/list",
      "/key/info",
      "/key/update",
      "/key/delete",
      "/key/regenerate",
      "/key/{key_id}/regenerate",
      "/key/aliases"
    ]
  }'
```

### Add the academic as team member

```bash
curl -ksS "$PROXY/team/member_add" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "team_id": "PUT-TEAM-ID-HERE",
    "member": {
      "role": "user",
      "user_id": "PUT-ACADEMIC-USER-ID-HERE"
    }
  }'
```

### Invite the academic

```bash
curl -ksS "$PROXY/invitation/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "user_id": "PUT-ACADEMIC-USER-ID-HERE"
  }'
```

Send them:

```text
https://gpustack.ing.unibs.it/ui/onboarding?invitation_id=<invitation-id>
```

Expected behavior:

- the academic can work in the UI
- the academic can create and manage API keys for their team
- their team is limited to the standard model allowlist
- their team has no budget limit by default
- they cannot use the restricted security and experimental model aliases unless you explicitly add those models to their team

If an academic needs multiple teams, prefer creating those teams centrally and adding the academic as a member with key-management permissions. Only use `proxy_admin` for people who should be trusted as platform operators.

## 9. Course student workflow

For normal courses, create one team per course or course group.

Example:

```bash
curl -ksS "$PROXY/team/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "team_alias": "course-ai-2026-group-a",
    "models": ["qwen35-4b", "ministral3-3b", "phi4-mini", "embeddinggemma"],
    "max_budget": 25,
    "budget_duration": "30d",
    "team_member_budget": 2
  }'
```

Students in normal courses should usually be:

- UI role: `internal_user_viewer`
- team member role: `user`
- model access: standard models only

They should not receive:

- `llamaguard3-1b`
- `granite-guardian-2b`
- `experimental-heretic-llama32-3b`
- `experimental-abliterated-llama32-3b`

## 10. Computer Security course workflow

For Computer Security groups, create a separate team with explicit access to the security models.

Example:

```bash
curl -ksS "$PROXY/team/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "team_alias": "course-computer-security-2026",
    "models": [
      "qwen35-4b",
      "ministral3-3b",
      "phi4-mini",
      "embeddinggemma",
      "llamaguard3-1b",
      "granite-guardian-2b",
      "experimental-heretic-llama32-3b",
      "experimental-abliterated-llama32-3b"
    ],
    "max_budget": 25,
    "budget_duration": "30d",
    "team_member_budget": 2
  }'
```

Recommended student role:

- use `internal_user_viewer` if they only need to inspect spend and use centrally generated keys
- use `internal_user` only if they should create their own keys

Recommended operating rule:

- keep Computer Security students in a separate team
- do not mix standard course students and security students in the same team
- do not add `experimental-heretic-llama32-3b` or `experimental-abliterated-llama32-3b` to any standard course team

## 11. Can students get UI access with limited functionality?

Yes, with caveats.

### What is possible now

You can give students UI access and limit them to mostly read-only behavior.

The practical options are:

- `internal_user_viewer`
  - can log in
  - can view their own keys
  - can view their own spend
  - cannot create/delete keys
  - cannot add users

- `internal_user`
  - can log in
  - can view their own spend
  - can create/delete their own keys

If your requirement is:

- "students can log into the UI and mainly check budget / usage"

then `internal_user_viewer` is the closest fit.

### Important caveat

LiteLLM documents `internal_user_viewer` as deprecated in favor of team/org-specific roles. However, it is still documented and is the most suitable role for your exact requirement today.

So the pragmatic answer is:

- yes, you can accommodate this now
- for your current deployment, `internal_user_viewer` is the simplest choice
- if later you want richer team governance, SSO, or more granular org/team roles, that moves toward the enterprise feature set

### UI visibility restriction

LiteLLM also supports hiding pages from internal users in the web UI.

That means you can make the student UI much narrower, for example showing only:

- Usage
- Budgets
- optionally Virtual Keys

This is configured either:

- in the UI: `Settings -> Admin Settings -> UI Settings -> Configure Page Visibility`
- or via API:
  - `GET /ui_settings/get`
  - `PATCH /ui_settings/update`

This is the main mechanism for giving students a simpler interface.

## 12. Day-to-day admin shell

The simplest way to administer the gateway is:

```bash
ssh litellm@bia001
cd /home/litellm/bia-litellm
set -a
. ./.env
set +a
export PROXY=https://gpustack.ing.unibs.it
```

After that, admin API calls can use:

- `Authorization: Bearer $LITELLM_MASTER_KEY`

## 13. Automation scripts

This repository includes scripts for the common access-management flows:

- `admin-scripts/create_academic_team.py`
- `admin-scripts/create_student_group.py`
- `admin-scripts/add_students_to_group.py`

They use the same LiteLLM API calls documented below, but with safer defaults and local JSON records under `access-records/`.

Read the script guide:

```bash
sed -n '1,220p' admin-scripts/README.md
```

Examples:

```bash
python3 admin-scripts/create_academic_team.py \
  --email colleague@example.org \
  --team-alias academic-colleague-lab
```

```bash
python3 admin-scripts/create_student_group.py \
  --group course-ai-2026-group-a
```

```bash
python3 admin-scripts/create_student_group.py \
  --group course-computer-security-2026 \
  --full-visibility
```

```bash
python3 admin-scripts/add_students_to_group.py \
  --group course-ai-2026-group-a \
  --emails-file students.txt \
  --generate-key \
  --key-budget 2
```

Use `--dry-run` on any script to inspect planned API calls without changing LiteLLM.

## 14. How to create a new UI user

Recommended for a student who should access the UI:

```bash
curl -ksS "$PROXY/user/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "user_email": "student1@example.org",
    "user_role": "internal_user_viewer"
  }'
```

Notes:

- use `internal_user_viewer` for read-only UI access
- use `internal_user` only if the student should be able to create their own keys

Expected result:

- LiteLLM returns a `user_id`
- keep that `user_id`, because you need it for invitations and team membership

## 15. How to invite that user

After creating the user:

```bash
curl -ksS "$PROXY/invitation/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "user_id": "PUT-USER-ID-HERE"
  }'
```

The response contains an invitation `id`.

The onboarding URL format is:

```text
https://gpustack.ing.unibs.it/ui/onboarding?invitation_id=<invitation-id>
```

This is the link to send to the student.

Practical rule:

- if you need a student-specific UI account, do not share a global admin credential
- always use user creation + invitation

## 16. How to handle student passwords

Recommended policy:

- do not set or share one common password for all students
- let each student complete the invitation/onboarding flow and use their own credentials
- if a student loses access, generate a new invitation link

Avoid using `UI_USERNAME` / `UI_PASSWORD` as the student mechanism. That setting is global and shared.

## 17. How to create a team for a class

Example:

```bash
curl -ksS "$PROXY/team/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "team_alias": "students-2026-lab-a",
    "models": ["qwen35-4b", "ministral3-3b", "phi4-mini", "embeddinggemma"],
    "max_budget": 25,
    "budget_duration": "30d",
    "team_member_budget": 2
  }'
```

Suggested interpretation:

- `max_budget`: shared ceiling for the whole class team
- `team_member_budget`: default per-student ceiling inside that team

For a classroom, this is usually better than managing only per-key budgets.

## 18. How to add a student to a team

If the user already exists:

```bash
curl -ksS "$PROXY/team/member_add" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "team_id": "PUT-TEAM-ID-HERE",
    "max_budget_in_team": 2,
    "member": {
      "role": "user",
      "user_id": "PUT-USER-ID-HERE"
    }
  }'
```

Use this when:

- you want the student budget enforced within the class team
- you want team-scoped keys

## 19. How to create an API key for a student

If you want a key tied to a specific student and team:

```bash
curl -ksS "$PROXY/key/generate" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "user_id": "PUT-USER-ID-HERE",
    "team_id": "PUT-TEAM-ID-HERE",
    "models": ["qwen35-4b", "ministral3-3b", "phi4-mini", "embeddinggemma"],
    "max_budget": 2,
    "budget_duration": "30d"
  }'
```

For students, do not include:

- `experimental-heretic-llama32-3b`
- `experimental-abliterated-llama32-3b`

Those should remain instructor-only.

## 20. How to check spend

### Per key

```bash
curl -ksS "$PROXY/key/info?key=PUT-KEY-HERE" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

### Per user

```bash
curl -ksS "$PROXY/user/info?user_id=PUT-USER-ID-HERE" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

### Per team

```bash
curl -ksS "$PROXY/team/info?team_id=PUT-TEAM-ID-HERE" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

## 21. How to block or restore a key

Temporarily block:

```bash
curl -ksS "$PROXY/key/block" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "key": "PUT-KEY-HERE"
  }'
```

Unblock:

```bash
curl -ksS "$PROXY/key/unblock" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "key": "PUT-KEY-HERE"
  }'
```

Use block/unblock before deleting anything. It is safer for classroom operations.

## 22. Recommended configuration for student UI

If you want students to log in and mostly just inspect budget/usage:

### Recommended role

- `internal_user_viewer`

### Recommended UI visibility

Make visible only:

- Usage
- Budgets
- optionally Virtual Keys

Hide:

- Models + Endpoints
- Teams
- Organizations
- Logs
- Agents
- MCP tools
- admin/config pages

### Recommended budget model

- one team per class/lab
- team-wide budget
- per-student `max_budget_in_team`
- student keys tied to both `user_id` and `team_id`

This gives you:

- individual accountability
- class budget control
- low-risk UI exposure

## 23. What I would do in practice

For a normal student lab, I would use this policy:

1. Create one team per course instance.
2. Put only student-safe models on the team.
3. Create each student as `internal_user_viewer`.
4. Invite each student through `/invitation/new`.
5. Restrict UI page visibility to `Usage` and `Budgets`, plus `Virtual Keys` only if you want them to see their key inventory.
6. Generate keys centrally as admin, instead of letting students create keys themselves.
7. Keep experimental models only on instructor-owned keys.

If later you want students to self-issue keys, switch selected users from `internal_user_viewer` to `internal_user`.

For fellow academics, I would use this policy:

1. Create one team per academic, lab, or course they own.
2. Create them as `internal_user`.
3. Add them as a normal team member with key-management permissions.
4. Keep their team on the standard model allowlist unless you explicitly approve security models.
5. Use `proxy_admin` only for people who should be platform operators.

For Computer Security courses, I would use this policy:

1. Create a dedicated Computer Security team.
2. Add only Computer Security students and instructors.
3. Add the security and experimental model aliases only to that team.
4. Use stricter budgets and shorter `budget_duration` for experimental access.
5. Keep logs and class rosters aligned so experimental use is attributable.

## 24. Operational caveats

- The current deployment does not yet have a documented per-user password reset flow in this repository.
- The safest recovery path is invitation re-issue.
- `internal_user_viewer` is documented but marked deprecated by LiteLLM.
- More granular org/team RBAC and some advanced governance features are tied to enterprise features.
- The simplest stable classroom setup is still fully workable with:
  - internal users
  - invitations
  - team budgets
  - scoped keys
  - page visibility restrictions

## 25. References

Official LiteLLM docs used for these operational notes:

- Admin UI quick start: `https://docs.litellm.ai/docs/proxy/ui`
- Virtual keys: `https://docs.litellm.ai/docs/proxy/virtual_keys`
- Budgets and rate limits: `https://docs.litellm.ai/docs/proxy/users`
- RBAC: `https://docs.litellm.ai/docs/proxy/access_control`
- Internal user self-serve: `https://docs.litellm.ai/docs/proxy/self_serve`
- UI page visibility: `https://docs.litellm.ai/docs/proxy/ui/page_visibility`
- Admin UI SSO: `https://docs.litellm.ai/docs/proxy/admin_ui_sso`
