# 4grade-graduateproject-melomatch

📜 Third-Party Licenses

이 프로젝트는 다음과 같은 오픈소스 라이브러리를 사용하며, 각 라이브러리는 고유의 라이선스를 따른다.
| Library                          | License            | 사용 목적                                     |
| -------------------------------- | ------------------ | ----------------------------------------- |
| **DeepFace**                     | MIT                | 얼굴 인식·나이·성별 추정 및 감정 분석                    |
| **MTCNN**                        | MIT                | 실시간 얼굴 검출(DeepFace 백엔드)                   |
| **transformers (HuggingFace)**   | Apache License 2.0 | KcELECTRA, Wav2Vec2 모델 로딩 및 파인튜닝          |
| **PyTorch / Torchaudio**         | BSD-style          | 모델 학습 및 추론                                |
| **google-cloud-speech**          | Apache License 2.0 | Google Speech-to-Text API를 통한 음성 → 텍스트 변환 |
| **OpenCV**                       | Apache License 2.0 | 실시간 영상 처리 및 카메라 제어                        |
| **PyQt5**                        | GPL/LGPL           | 데스크톱 GUI 제작                               |
| **selenium / webdriver-manager** | Apache License 2.0 | 유튜브 자동 검색 및 음악 재생 제어                      |
| **imutils**                      | MIT                | 영상 전처리 유틸리티                               |
| **midiutil / pygame**            | MIT                | 감정 기반 실시간 MIDI 작곡 및 재생                    |

프로젝트 설명

Melomatch는 카메라와 마이크를 통해 수집된 영상·음성 데이터를 동시에 분석하여 사용자의 현재 감정 상태를 추론하고 
그 결과에 따라 유튜브 음악을 추천하거나 AI가 직접 작곡한 음악을 실시간으로 재생하는 멀티모달 감정 인식 기반 음악 추천 시스템이다.

🔎 동작 개요

데이터 입력 : 웹캠과 마이크로부터 실시간 영상과 음성을 수집

텍스트 변환 : Google Speech-to-Text API로 음성을 텍스트로 변환

감정 분석

DeepFace + MTCNN : 얼굴 이미지에서 성별·나이·표정 감정 추출

Wav2Vec2 : 음성의 높이·세기 등 음향적 특징으로 감정 분류

KcELECTRA : 텍스트 기반 7가지 감정(angry, happy, sad, neutral, surprise, fear, disgust) 분류

멀티모달 결합 : 텍스트 0.8 + 음성 0.2 가중치로 최종 감정 산출

음악 생성/추천

감정 결과가 안정적이면 유튜브 API + Selenium을 통해 관련 음악 자동 검색·재생

사용자가 AI 작곡 모드를 선택하면 MIDIUtil로 템포·스케일·악기 조합을 결정하고 Pygame으로 실시간 연주

GUI 시현 : PyQt5로 제작한 데스크톱 프로그램에서 카메라 영상, 분석 결과, 추천 음악 제어 버튼 등을 실시간 표시

📊 모델 학습

텍스트 감정 모델(KcELECTRA) : AI Hub 한국어 대화 음성 텍스트 → 7감정 분류

음성 감정 모델(Wav2Vec2) : AI Hub 한국어 대화 음성의 스펙트로그램·음향 특징을 입력으로 학습

얼굴 감정/성별/나이 모델 : UTKFace 등 공개 데이터셋 활용

학습 후 .pt 모델을 src/models/ 폴더에 저장하고, final_gui.py가 로드하여 실시간 추론을 수행

⚙️ 개발 환경

개발 도구 : PyCharm 2023 / Google Colab (학습)

실행 환경 : Python 3.10, GPU 학습 지원(Colab)

GUI 실행 : pip install -r requirements.txt
python src/final_gui.py


