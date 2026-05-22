# -*- coding: utf-8 -*-
"""Pytest 공통 설정.

각 테스트 파일이 `from activations import ReLU`처럼 `src/` 아래 모듈을
직접 import할 수 있도록 프로젝트 루트의 `src/` 폴더를 import 경로에 추가합니다.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
