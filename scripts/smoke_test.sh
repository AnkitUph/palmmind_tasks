#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-http://localhost:8000}"
SESSION="smoke-$(date +%s)"
BOOK_DATE="$(date -d '+3 days' +%F)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
post_chat() {
  curl -sS --max-time 360 -X POST "$BASE/chat" -H 'Content-Type: application/json' \
    -d "{\"session_id\": \"$SESSION\", \"message\": \"$1\"}" \
    -w '\n[HTTP %{http_code}, %{time_total}s]\n'
}

wait_for_api() {
  for attempt in $(seq 1 90); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
      echo "API is ready."
      return 0
    fi
    sleep 2
  done
  echo "The API did not become ready within 3 minutes."
  echo "Check the logs with: docker compose logs api --tail 50"
  return 1
}

cat > "$TMP/handbook.txt" <<'DOC'
The office opens at 9am and closes at 5pm, Monday to Friday.
Employees get 20 paid leave days per year. Unused leave days carry over for up to 5 days.
Remote work is allowed on Fridays. Equipment for remote work is provided by the company.
DOC

step "0. Waiting for the API to finish starting (it loads the embedding model)"
wait_for_api

step "1. Health check"
curl -sS "$BASE/health"; echo

step "2. Ingest with strategy=fixed"
curl -sS -X POST "$BASE/ingest" -F "file=@$TMP/handbook.txt" -F "strategy=fixed"; echo

step "3. Ingest with strategy=sentence"
curl -sS -X POST "$BASE/ingest" -F "file=@$TMP/handbook.txt" -F "strategy=sentence"; echo

step "4. Wrong file type should return 415"
echo "x" > "$TMP/bad.png"
curl -sS -o /dev/null -w "HTTP %{http_code}\n" -X POST "$BASE/ingest" -F "file=@$TMP/bad.png"

step "5. Question answered from the document"
post_chat "When does the office open?"

step "6. Follow-up in the same session (multi-turn)"
post_chat "And what about working remotely, which day is allowed?"

step "7. Question the documents cannot answer"
post_chat "What is the price of a Tesla Model 3?"

step "8. Interview booking, turn 1 (name)"
post_chat "I want to book an interview. My name is Ankit Sharma."

step "9. Interview booking, turn 2 (email)"
post_chat "My email is ankit@example.com"

step "10. Interview booking, turn 3 (date and time) -> should confirm"
post_chat "Please book it for $BOOK_DATE at 15:30"

step "11. Bookings saved in Postgres (needs Docker; skipped otherwise)"
if docker compose exec -T postgres psql -U rag -d rag -c "select id, name, email, interview_date, interview_time from bookings order by id desc limit 3;" 2>/dev/null; then
  :
else
  echo "Skipped. Check the 'bookings' table in your database yourself."
fi

printf '\nDone. Read each answer above and check it makes sense.\n'
