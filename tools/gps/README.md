# GPS 경로 재구성 (`tools/gps`)

유튜브 공개 주행 영상에는 GPS가 없어서 경로를 다시 만든다 (PRD 8.5, docs/plan.md 0.7·0.8).
결과는 가상 블랙박스가 읽는 `t_s,lat,lon,speed_kmh` CSV (`contracts` 의 `GpsTrack`)이다.

| 파일 | 하는 일 |
|---|---|
| `routes/<video_id>.gpx` | gpx.studio에서 도로를 따라 그린 경로 (주행 방향 순서) |
| `anchors/<video_id>.csv` | 기준점: `video_id,t_s,lat,lon,memo` |
| `reconstruct.py` | 기준점을 경로 위 가장 가까운 점에 붙이고, 기준점 사이를 **경로를 따라 거리 비례**로 보간 → `data/videos/<video_id>.gps.csv` (0.5초 간격) |
| `holdout_error.py` | 기준점을 하나씩 빼고 보간 위치와 실제 위치의 거리(m) → `experiments/gps_error.csv` (영상별 평균·중앙값·최대) |
| `preview_map.py` | 경로·기준점·재구성 점·hold-out 위치를 인터넷 없이 열리는 HTML(SVG)로 그린다. 배경은 `data/basemap/` 도로선 |

`gn_a`·`gn_b` 양식(빈 GPX, 헤더만 있는 기준점 CSV)이 들어 있다. **기준점 시각은 영상을 보고 사람이 정한다.**

## 순서

1. **경로 그리기** — [gpx.studio](https://gpx.studio)에서 자동차 경로 모드로 시작점→끝점을 클릭해 도로에 맞춘다.
   - 영상과 **같은 방향**(gn_a·gn_b 모두 강남대로 강남역→신논현역 **북행**)으로, 중앙분리대가 있으면 북행 쪽 차로에 그린다.
   - 첫 기준점보다 조금 앞에서 시작해 마지막 기준점보다 조금 뒤에서 끝낸다. 같은 구간이면 GPX 하나를 두 영상에 복사해도 된다.
   - GPX로 내보내 `routes/<video_id>.gpx` 를 바꾼다.
2. **기준점 고르기** — 영상당 **8개 이상**, 구간 처음과 끝에 하나씩 넣고 그 사이를 고르게 채운다.
   - 좋은 기준점: 교차로 정지선·횡단보도를 지나는 순간, 육교·고가 아래로 들어가는 순간, 큰 도로표지판 바로 아래, 지하철 출입구·큰 건물 입구 옆. 지도에서 정확한 위치를 찍을 수 있는 곳.
   - 시각 `t_s`: **원본 유튜브 영상 시각**. 초(`1250.5`)나 분:초(`20:50.5`). 멈춘 화면에서 `,` `.` 키로 한 프레임씩 움직여 그 지점을 지나는 순간을 고른다.
   - 위치 `lat,lon`: gpx.studio(또는 지도)에서 그 지점의 **주행 차로 위**를 찍어 소수 6자리 이상. 경로에서 50 m 넘게 떨어지면 오류(`--max-snap-m`).
   - **신호 대기·정체로 멈춘 구간**: 멈춘 순간과 다시 출발한 순간을 둘 다 적는다. 출발 순간 줄은 `lat,lon` 을 비운다 (= 앞 기준점과 같은 곳). 멈춤을 빼먹으면 그 전후가 평균 속도로 퍼져 오차가 커진다 (gn_a는 정체가 잦다).
   - 시간 순으로 적는다. 시간이 거꾸로이거나 경로 순서와 어긋나면 오류. `#` 로 시작하는 줄은 메모로 무시된다.
3. **만들고 확인하기**

   ```bash
   uv run python -m tools.gps.reconstruct gn_a                      # → data/videos/gn_a.gps.csv (+ 구간별 거리·속도)
   uv run python -m tools.gps.preview_map gn_a --fetch-roads        # 처음 한 번만 --fetch-roads (인터넷, OSM). 이후 오프라인
   uv run python -m tools.gps.holdout_error gn_a gn_b               # → experiments/gps_error.csv
   ```

   - `reconstruct` 가 구간 평균 속도를 보여 준다. 120 km/h 를 넘는 구간이나 0 km/h 가 아닌데 영상에서 멈춰 있던 구간은 기준점을 다시 본다.
   - `preview_map` 의 HTML을 열어 파란 경로가 회색 도로 위에 있는지, 빨간 기준점(번호)이 맞는 교차로에 있는지 본다. 휠로 확대한다. 받은 도로(`data/basemap/gps_<id>.json`)는 관리자 화면의 오프라인 도로선 지도에도 나온다.
4. **클립에 맞추기** — `<id>.mp4`·`<id>_1080p30.mp4` 는 원본과 시각이 같아 `gn_a.gps.csv` 를 그대로 `--gps` 로 준다.
   잘라 낸 클립(`<id>_MMSS-MMSS.mp4`)은 t=0 이 원본의 다른 시각이므로 `tools/video` 의 `fetch_log.jsonl` 에 있는 `clip_start_in_source_s` 를 빼서 영상 옆 이름으로 만든다:

   ```bash
   uv run python -m tools.gps.reconstruct gn_a --clip-start <clip_start_in_source_s> --out data/videos/gn_a_2050-2510.gps.csv
   ```

CSV는 첫 기준점부터 마지막 기준점까지만 있다. 그 밖의 프레임은 가상 블랙박스가 위치 없음으로 건너뛴다.

## 오차 읽기 (AC-13)

- hold-out은 기준점 하나를 빼서 간격이 두 배가 된 상태로 재므로, 모든 기준점을 쓴 실제 재구성보다 **크게(보수적으로)** 나온다.
- 멈춘 구간의 시작·끝 기준점을 빼면 그 멈춤을 모르는 채로 보간하므로 오차가 특히 크다. 최대값보다 평균·중앙값과 `max_at` 을 같이 본다.
- 평균이 20 m(중복 묶기 반경 시작값)를 넘으면 기준점을 더 넣는다 (docs/plan.md 리스크 표).
- 거리는 기준점에 적은 위치(찍은 곳)와의 직선 거리라서, 찍을 때 생긴 오차(`preview_map` 표의 "경로까지 m")도 들어 있다.
