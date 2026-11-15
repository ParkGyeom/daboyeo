#!/usr/bin/env bash
# 시연 환경을 한 번에 띄우고 되돌린다 (docs/DESIGN.md 9장).
#
#   scripts/demo.sh up       DB(Docker) → 서버(8000) → 화면(8501) 을 띄운다
#   scripts/demo.sh status   무엇이 떠 있는지 본다
#   scripts/demo.sh reset    DB 와 사진을 비워 처음 상태로 (시연용 시민 3명은 다시 넣는다)
#   scripts/demo.sh drive <모델.onnx> [기기 수] [구간] [배속]
#                            가상 블랙박스를 기기 수만큼 동시에 돌린다 (기본 1대, 서울 우천 구간, 4배속)
#                            2대째부터는 GPS 오차·밝기·시작 시점을 달리한다
#   scripts/demo.sh down     서버와 화면을 끈다 (DB 는 둔다. DB 까지 끄려면 down --db)
#
# 로그는 server_data/logs/ 에 남는다.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=server_data/run
LOGS=server_data/logs
SERVER_PORT=${SERVER_PORT:-8000}
WEB_PORT=${WEB_PORT:-8501}
DEFAULT_SEGMENT=0401_light_rainy_smooth:03000071-03000182
mkdir -p "$RUN" "$LOGS"

say() { printf '%s\n' "$*"; }
die() { printf '오류: %s\n' "$*" >&2; exit 1; }

is_running() {  # $1 = 이름
    local pidfile="$RUN/$1.pid"
    [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_http() {  # $1 = 주소, $2 = 이름
    for _ in $(seq 1 60); do
        curl -fs -o /dev/null "$1" && return 0
        sleep 1
    done
    die "$2 이(가) 60초 안에 뜨지 않았다. $LOGS/$2.log 를 본다"
}

start_bg() {  # $1 = 이름, 나머지 = 명령
    local name=$1; shift
    nohup "$@" >"$LOGS/$name.log" 2>&1 &
    echo $! >"$RUN/$name.pid"
}

cmd_up() {
    [[ -x $PY ]] || die ".venv 가 없다. CLAUDE.md 의 환경 설정을 먼저 한다"
    docker info >/dev/null 2>&1 || die "Docker 가 꺼져 있다. Docker Desktop 을 켠 뒤 다시 실행한다"

    say "DB 띄우는 중..."
    docker compose up -d db >/dev/null 2>&1
    for _ in $(seq 1 60); do
        docker compose ps db 2>/dev/null | grep -q "(healthy)" && break
        sleep 2
    done
    docker compose ps db | grep -q "(healthy)" || die "DB 가 준비되지 않았다 (docker compose logs db)"
    say "  DB 준비됨 (5432)"

    if is_running server; then
        say "  서버는 이미 떠 있다"
    else
        port_busy "$SERVER_PORT" && die "포트 $SERVER_PORT 을(를) 다른 프로그램이 쓰고 있다"
        start_bg server "$PY" -m uvicorn server.main:app --port "$SERVER_PORT"
        wait_http "http://localhost:$SERVER_PORT/health" server
        say "  서버 준비됨 http://localhost:$SERVER_PORT"
    fi

    if is_running web; then
        say "  화면은 이미 떠 있다"
    else
        port_busy "$WEB_PORT" && die "포트 $WEB_PORT 을(를) 다른 프로그램이 쓰고 있다"
        SERVER_URL="http://localhost:$SERVER_PORT" start_bg web .venv/bin/streamlit run ui/app.py \
            --server.port "$WEB_PORT" --server.headless true --browser.gatherUsageStats false
        wait_http "http://localhost:$WEB_PORT/_stcore/health" web
        say "  화면 준비됨 http://localhost:$WEB_PORT"
    fi
}

cmd_down() {
    for name in web server; do
        if is_running "$name"; then
            kill "$(cat "$RUN/$name.pid")" 2>/dev/null || true
            say "$name 끔"
        fi
        rm -f "$RUN/$name.pid"
    done
    pkill -f "edge.blackbox.run" 2>/dev/null && say "가상 블랙박스 끔" || true
    if [[ ${1:-} == --db ]]; then
        docker compose stop db >/dev/null && say "DB 끔"
    fi
}

cmd_status() {
    if docker info >/dev/null 2>&1 && docker compose ps db 2>/dev/null | grep -q "(healthy)"; then
        say "DB     켜짐"
    else
        say "DB     꺼짐"
    fi
    for name in server web; do
        if is_running "$name"; then say "$(printf '%-6s' "$name") 켜짐 (pid $(cat "$RUN/$name.pid"))"; else say "$(printf '%-6s' "$name") 꺼짐"; fi
    done
    local n
    n=$({ pgrep -f "edge.blackbox.run" 2>/dev/null || true; } | wc -l | tr -d ' ')
    say "블랙박스 ${n}대 실행 중"
    if curl -fs "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
        curl -fs "http://localhost:$SERVER_PORT/api/defects" | "$PY" -c \
            'import json,sys; d=json.load(sys.stdin); n=sum(x["status"]=="미신고" for x in d); print(f"결함 {len(d)}건 (미신고 {n})")'
    fi
}

cmd_reset() {
    docker info >/dev/null 2>&1 || die "Docker 가 꺼져 있다"
    pgrep -f "edge.blackbox.run" >/dev/null 2>&1 && die "가상 블랙박스가 돌고 있다. 끝난 뒤에 되돌린다"
    "$PY" -c "from server import db; db.init_db(); db.reset_db()"
    rm -f server_data/snapshots/*.jpg
    say "처음 상태로 되돌렸다 (결함·신고·사진 삭제, 시민 3명 다시 넣음)"
}

cmd_drive() {
    local model=${1:-}; local devices=${2:-1}; local segment=${3:-$DEFAULT_SEGMENT}; local speed=${4:-4}
    [[ -n $model && -f $model ]] || die "모델 파일을 준다: scripts/demo.sh drive runs/<모델>.onnx [기기 수] [구간] [배속]"
    [[ $devices =~ ^[1-3]$ ]] || die "기기 수는 1~3 (시연용 시민이 3명이다)"
    curl -fs "http://localhost:$SERVER_PORT/health" >/dev/null || die "서버가 꺼져 있다. scripts/demo.sh up 을 먼저 한다"

    # 기기별 조건(GPS 오차·밝기·시작 시점)은 edge/blackbox/variants.py 한 곳에 있다
    "$PY" -m edge.blackbox.variants --devices "$devices" --segment "$segment" --model "$model" --speed "$speed" \
        --server "http://localhost:$SERVER_PORT" 2>&1 | tee "$LOGS/blackbox.log"
}

case ${1:-} in
    up) cmd_up ;;
    down) shift; cmd_down "${1:-}" ;;
    status) cmd_status ;;
    reset) cmd_reset ;;
    drive) shift; cmd_drive "$@" ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
