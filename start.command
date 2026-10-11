#!/bin/bash
# 더블클릭으로 실행: 첫 실행이면 필요한 모든 프로그램을 설치한 뒤 ElementarySchoolForLLM을 시작합니다.
cd "$(dirname "$0")" || exit 1
exec /bin/bash ./bootstrap.sh ${@+"$@"}
