#!/usr/bin/env bash
# One autonomous wiki tick: generate the next UNCOVERED topic, commit, push.
# Breadth-first only — never re-generates a 'done' topic (no churn until --enrich).
#
# Manual:   ./autopilot.sh
# Cron:     */30 * * * * /home/gokul/research-wiki/autopilot.sh >> autopilot.log 2>&1
set -uo pipefail
cd /home/gokul/research-wiki

ts() { date -u +%FT%TZ; }

# pick the next uncovered topic (empty/draft before done); empty string if none
NEXT=$(.venv/bin/python -c "import yaml;t=yaml.safe_load(open('taxonomy.yaml'));u=[x['slug'] for x in t['topics'] if x.get('status')!='done'];print(u[0] if u else '')")
if [ -z "$NEXT" ]; then
  echo "$(ts) queue empty — all topics done, nothing to do"; exit 0
fi

# require the local model server (orchestrator/distill) to be up
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8181/health)" != "200" ]; then
  echo "$(ts) qwen (:8181) not ready — skipping tick"; exit 1
fi

echo "$(ts) generating: $NEXT"
if ! .venv/bin/python agent.py "$NEXT"; then
  echo "$(ts) generation failed for $NEXT"; exit 1
fi

git add -A
if git commit -q -m "auto: $NEXT ($(ts))"; then
  if git remote get-url origin >/dev/null 2>&1; then
    git push -q origin "$(git branch --show-current)" && echo "$(ts) pushed $NEXT" \
      || echo "$(ts) push failed"
  else
    echo "$(ts) committed $NEXT (no remote configured yet)"
  fi
fi
echo "$(ts) done: $NEXT"
