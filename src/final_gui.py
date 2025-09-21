import os
import sys
import subprocess
import cv2
import pygame
import torch
import random
import json
import numpy as np
import pyaudio
import wave
import warnings
from transformers import AutoTokenizer, Wav2Vec2ForSequenceClassification, AutoModelForSequenceClassification
from mtcnn import MTCNN
from google.cloud import speech
from pydub import AudioSegment
from pygame import mixer
import time
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import cv2
import numpy as np
from datetime import datetime
import torchaudio
from deepface import DeepFace
from midiutil import MIDIFile
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 읽기
API_KEY = os.getenv("API_KEY")
BASE_PATH = os.getenv("BASE_PATH")
WEIGHT_FILE = os.getenv("WEIGHT_FILE", "weights.json")


class KcELECTRAWithDropout(torch.nn.Module):
    def __init__(self):
        super(KcELECTRAWithDropout, self).__init__()
        self.electra = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", num_labels=7)
        self.dropout = torch.nn.Dropout(p=0.6)
        self.batch_norm = torch.nn.BatchNorm1d(7)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # ElectraForSequenceClassification은 token_type_ids를 자동으로 처리합니다
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.dropout(outputs.logits)
        return self.batch_norm(logits)

    def load_state_dict(self, state_dict, strict=True):
        try:
            # weights_only=True를 사용하여 안전하게 로드
            return super().load_state_dict(state_dict, strict=strict)
        except Exception as e:
            print(f"모델 가중치 로드 중 오류 발생: {e}")
            raise


class EmotionAnalyzer:
    def __init__(self):
        try:
            # GPU 메모리 정리 및 제한
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.5)  # 메모리 사용량을 50%로 제한
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            print("GPU 초기화 실패, CPU를 사용합니다.")
            self.device = torch.device('cpu')

        # 기본 가중치 정의
        self.default_weights = {
            "화남": 1.0,
            "행복": 1.0,
            "슬픔": 1.0,
            "중립": 1.0,
            "놀람": 1.0,
            "공포": 1.0,
            "혐오": 1.0
        }
        self.weights = self.load_weights()

        # YouTube API 초기화
        self.youtube = build('youtube', 'v3', developerKey=API_KEY)

        # 모델 초기화
        self.init_audio_system()
        self.init_models()
        self.init_music_system()

    def init_audio_system(self):
        """오디오 시스템 초기화"""
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()

    def init_music_system(self):
        """음악 시스템 초기화"""
        self.emotion_instruments = {
            "행복": {"instruments": [0, 24, 73, 40, 105], "tempo": (120, 140), "scale": "major"},
            "슬픔": {"instruments": [0, 42, 71, 48, 50], "tempo": (60, 80), "scale": "minor"},
            "화남": {"instruments": [30, 115, 33, 62, 56], "tempo": (140, 160), "scale": "minor"},
            "중립": {"instruments": [0, 25, 65, 42, 74], "tempo": (90, 110), "scale": "major"},
            "공포": {"instruments": [20, 41, 47, 89, 58], "tempo": (100, 120), "scale": "minor"},
            "놀람": {"instruments": [73, 81, 40, 72, 123], "tempo": (110, 130), "scale": "major"},
            "혐오": {"instruments": [43, 34, 58, 47, 55], "tempo": (80, 100), "scale": "minor"}
        }

    def init_models(self):
        """모델 초기화"""
        try:
            print(f"Using device: {self.device}")

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(BASE_PATH, "STT.json")
            audio_model_path = os.path.join(BASE_PATH, "수정음성_model.pt")
            text_model_path = os.path.join(BASE_PATH, "Text_model.pth")

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        "kresnik/wav2vec2-large-xlsr-korean",
                        num_labels=7,
                        ignore_mismatched_sizes=True
                    ).to('cpu')

                # 원래 구조대로 classifier 수정
                self.audio_model.classifier = torch.nn.Linear(1024, 7)
                # projector는 사용하지 않도록 설정
                self.audio_model.projector = torch.nn.Identity()

                audio_state_dict = torch.load(audio_model_path, map_location='cpu')
                new_state_dict = {k.replace('base_model.', ''): v for k, v in audio_state_dict.items()}

                # classifier 가중치는 제외하고 로드
                classifier_keys = [k for k in new_state_dict.keys() if 'classifier' in k]
                for k in classifier_keys:
                    del new_state_dict[k]

                self.audio_model.load_state_dict(new_state_dict, strict=False)
                self.audio_model.eval()

                if self.device.type == 'cuda':
                    self.audio_model = self.audio_model.to(self.device)
            except RuntimeError as e:
                print(f"오디오 모델 GPU 로드 실패, CPU 사용: {e}")
                self.device = torch.device('cpu')

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # 텍스트 토크나이저 초기화
                    self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

                    # 텍스트 모델 초기화
                    self.text_model = KcELECTRAWithDropout().to('cpu')
                    # weights_only 옵션 제거
                    text_state_dict = torch.load(text_model_path, map_location='cpu')
                    self.text_model.load_state_dict(text_state_dict, strict=True)
                    self.text_model.eval()

                    if self.device.type == 'cuda':
                        self.text_model = self.text_model.to(self.device)
            except RuntimeError as e:
                print(f"텍스트 모델 GPU 로드 실패, CPU 사용: {e}")
                self.device = torch.device('cpu')
                if hasattr(self, 'text_model'):
                    self.text_model = self.text_model.to('cpu')

            # MTCNN 얼굴 감지기 초기화
            try:
                self.detector = MTCNN()
            except Exception as e:
                print(f"MTCNN 초기화 실패: {e}")
                raise

        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            raise

    def capture_image_until_face(self):
        """얼굴이 인식될 때까지 웹캠 활성화하고 정보 표시"""
        print("\n웹캠을 활성화합니다. 얼굴이 인식될 때까지 기다립니다...")
        print("종료하려면 'q'를 누르세요")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다!")
            return None

        face_detected = False
        detected_frame = None
        face_info = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # 조도 조정 - 히스토그램 균등화 적용
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                equalized_frame = cv2.equalizeHist(gray_frame)
                display_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

                # 얼굴 감지
                faces = self.detector.detect_faces(display_frame)

                if faces and not face_detected:
                    print("얼굴이 인식되었습니다!")
                    face_detected = True
                    detected_frame = display_frame.copy()

                    # DeepFace 분석
                    face = faces[0]
                    x, y, w, h = face['box']
                    face_frame = frame[y:y + h, x:x + w]

                    try:
                        analysis = DeepFace.analyze(
                            face_frame,
                            actions=['age', 'gender'],
                            enforce_detection=False,
                            silent=True
                        )

                        if isinstance(analysis, list):
                            analysis = analysis[0]

                        # 성별 판단: 임계값 설정으로 보수적 판단
                        male_confidence = analysis['gender']['Man']
                        female_confidence = analysis['gender']['Woman']

                        if abs(male_confidence - female_confidence) < 10:  # 성별 구분이 애매한 경우 'Unknown'으로 설정
                            gender = 'Unknown'
                        else:
                            gender = 'Male' if male_confidence > female_confidence else 'Female'

                        age_group = self.classify_age(analysis['age'])
                        face_info = (gender, age_group)
                    except Exception as e:
                        print(f"얼굴 분석 중 오류: {e}")
                        continue

                # 얼굴 정보 표시
                if face_info:
                    gender, age_group = face_info
                    cv2.putText(display_frame, f"Gender: {gender}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Age Group: {age_group}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 얼굴 인식 표시
                if faces:
                    for face in faces:
                        x, y, w, h = face['box']
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('Webcam', display_frame)

                if face_detected and detected_frame is not None:
                    return detected_frame

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            return None
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def record_audio_stream(self):
        """실시간 오디오 스트림 시작"""
        print("\n마이크를 활성화합니다. 말씀하실 때 녹음이 시작됩니다...")
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            frames = []
            print("녹음을 시작합니다...")

            for _ in range(0, int(self.rate / self.chunk * 5)):  # 5초간 녹음
                data = stream.read(self.chunk)
                frames.append(data)

            print("녹음이 완료되었습니다.")

            # 파일 저장
            with wave.open('recorded_audio.wav', 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))

            return 'recorded_audio.wav'
        except Exception as e:
            print(f"오디오 스트림 처리 중 오류 발생: {e}")
            return None
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()

    def analyze_emotions(self):
        """감정 분석 실행"""
        frame = self.capture_image_until_face()
        if frame is None:
            return None

        try:
            faces = self.detector.detect_faces(frame)
            face = faces[0]
            x, y, w, h = face['box']
            face_frame = frame[y:y + h, x:x + w]

            face_analysis = DeepFace.analyze(face_frame,
                                             actions=['age', 'gender'],
                                             enforce_detection=False,
                                             silent=True)

            if isinstance(face_analysis, list):
                face_analysis = face_analysis[0]

            print("\n=== 얼굴 분석 결과 ===")
            print(f"감지된 성별: {face_analysis['gender']}")
            print(f"추정 나잇대: {face_analysis['age']}")

            audio_file = self.record_audio_stream()
            if audio_file is None:
                return None

            emotion = self.analyze_audio_emotion(audio_file)
            return {
                'gender': 'Male' if face_analysis['gender']['Man'] > face_analysis['gender']['Woman'] else 'Female',
                'age_group': self.classify_age(face_analysis['age']),
                'emotion': emotion
            }
        except Exception as e:
            print(f"감정 분석 중 오류 발생: {e}")
            return None

    def analyze_audio_emotion(self, audio_path):
        """오디오 감정 분석"""
        print("\n음성을 분석하고 있습니다...")
        try:
            with torch.no_grad():
                # 오디오 처리
                waveform, sample_rate = torchaudio.load(audio_path)

                # 스테레오를 모노로 변환
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # 샘플레이트 변환
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)

                # 데이터 정규화 및 크기 조정
                max_length = 16000 * 10  # 10초 길이로 제한
                if waveform.shape[-1] > max_length:
                    waveform = waveform[..., :max_length]

                # 패딩 추가
                if waveform.shape[-1] < max_length:
                    padding_length = max_length - waveform.shape[-1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding_length))

                # CPU에서 처리 후 GPU로 이동
                waveform = waveform.to('cpu')
                waveform = waveform.float()  # 명시적으로 float32로 변환

                # 배치 차원 추가
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                if len(waveform.shape) == 2:
                    waveform = waveform.unsqueeze(0)

                # 이제 GPU로 이동
                waveform = waveform.to(self.device)

                try:
                    # 더 작은 배치로 처리
                    audio_emotion = self.audio_model(waveform).logits
                except RuntimeError as e:
                    print("GPU 처리 실패, CPU로 전환합니다...")
                    # GPU 실패시 CPU로 폴백
                    waveform = waveform.cpu()
                    self.audio_model = self.audio_model.cpu()
                    audio_emotion = self.audio_model(waveform).logits
                    self.audio_model = self.audio_model.to(self.device)

                # STT 변환
                client = speech.SpeechClient()
                with open(audio_path, "rb") as audio_file:
                    content = audio_file.read()

                response = client.recognize(
                    config=speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        language_code="ko-KR"
                    ),
                    audio=speech.RecognitionAudio(content=content)
                )

                transcript = ""
                for result in response.results:
                    transcript = result.alternatives[0].transcript

                # 텍스트 감정 분석
                text_input = self.tokenizer(transcript, padding=True, truncation=True, return_tensors="pt")
                text_input = {k: v.to(self.device) for k, v in text_input.items()}

                try:
                    text_emotion = self.text_model(**text_input)
                except RuntimeError:
                    print("GPU 처리 실패, CPU로 전환합니다...")
                    # GPU 실패시 CPU로 폴백
                    text_input = {k: v.cpu() for k, v in text_input.items()}
                    self.text_model = self.text_model.cpu()
                    text_emotion = self.text_model(**text_input)
                    self.text_model = self.text_model.to(self.device)

                # 결과 결합
                combined_emotion = 0.9 * text_emotion + 0.1 * audio_emotion
                emotion_idx = torch.argmax(torch.softmax(combined_emotion, dim=1)).item()

                emotions = ["화남", "행복", "슬픔", "중립", "놀람", "공포", "혐오"]
                return emotions[emotion_idx]

        except Exception as e:
            print(f"오디오 감정 분석 중 오류 발생: {e}")
            # CPU로 전환 시도
            try:
                self.device = torch.device('cpu')
                self.audio_model = self.audio_model.cpu()
                self.text_model = self.text_model.cpu()
                return self.analyze_audio_emotion(audio_path)
            except Exception as e:
                print(f"CPU 처리 중에도 오류 발생: {e}")
                raise

    def create_music(self, emotion):
        """향상된 MIDI 음악 생성 - 더 긴 버전"""
        print(f"\n{emotion} 감정에 맞는 음악을 작곡하고 있습니다...")
        try:
            emotion_config = self.emotion_instruments[emotion]
            tempo = random.randint(*emotion_config["tempo"])
            scale_type = emotion_config["scale"]

            # 트랙 수 설정
            track_count = len(emotion_config["instruments"])
            midi = MIDIFile(track_count)

            # 음악 구조 정의 - 더 긴 길이로 수정
            measures = 32  # 마디 수를 32마디로 증가
            beats_per_measure = 4  # 4/4 박자
            total_beats = measures * beats_per_measure

            # 음계 정의
            scales = {
                "major": [0, 2, 4, 5, 7, 9, 11],  # C major scale
                "minor": [0, 2, 3, 5, 7, 8, 10]  # C minor scale
            }
            current_scale = scales[scale_type]

            # 섹션 구조 정의 (A-B-A 형식)
            section_a = total_beats // 2  # 전체의 절반
            section_b = total_beats // 4  # 전체의 1/4

            for track in range(track_count):
                # 트랙 기본 설정
                midi.addTempo(track, 0, tempo)
                midi.addProgramChange(track, track, 0, emotion_config["instruments"][track])

                # 트랙별 특성 설정
                if track == 0:  # 메인 멜로디
                    volume_range = (90, 100)
                    note_durations = [1, 0.5, 0.25, 2]  # 다양한 음표 길이
                    octave_range = (5, 6)
                else:  # 반주
                    volume_range = (60, 80)
                    note_durations = [0.5, 1, 2, 4]
                    octave_range = (3, 5)

                # A 섹션 생성
                time = 0
                melody_a = []  # A 섹션의 멜로디 저장
                while time < section_a:
                    duration = random.choice(note_durations)
                    if time + duration > section_a:
                        duration = section_a - time

                    note = random.choice(current_scale)
                    octave = random.randint(*octave_range)
                    pitch = note + (12 * octave)
                    volume = random.randint(*volume_range)

                    # 멜로디 저장 및 추가
                    melody_a.append((pitch, duration, volume))
                    midi.addNote(track, track, pitch, time, duration, volume)

                    # 화음 추가 (메인 멜로디의 경우)
                    if track == 0 and random.random() < 0.3:
                        harmony_note = (note + 4) % 12
                        harmony_pitch = harmony_note + (12 * octave)
                        midi.addNote(track, track, harmony_pitch, time, duration, volume - 20)

                    time += duration

                # B 섹션 생성 (변주)
                time = section_a
                while time < section_a + section_b:
                    duration = random.choice(note_durations)
                    if time + duration > section_a + section_b:
                        duration = section_a + section_b - time

                    note = random.choice(current_scale)
                    octave = random.randint(*octave_range)
                    pitch = note + (12 * octave)
                    volume = random.randint(*volume_range)

                    midi.addNote(track, track, pitch, time, duration, volume)
                    time += duration

                # A 섹션 반복 (처음 저장한 멜로디 재사용)
                time = section_a + section_b
                for pitch, duration, volume in melody_a:
                    if time + duration > total_beats:
                        break
                    midi.addNote(track, track, pitch, time, duration, volume)
                    time += duration

            # 파일명에 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{emotion}_{timestamp}.mid"

            with open(filename, "wb") as output_file:
                midi.writeFile(output_file)

            print(f"MIDI 파일이 생성되었습니다: {filename}")
            return filename

        except Exception as e:
            print(f"음악 생성 중 오류 발생: {e}")
            return None

    def play_midi_file(self, midi_file):
        """향상된 MIDI 파일 재생"""
        try:
            # pygame 완전히 초기화
            pygame.init()
            pygame.mixer.quit()  # 기존 mixer 초기화
            pygame.mixer.init(frequency=44100)

            if not os.path.exists(midi_file):
                print(f"MIDI 파일을 찾을 수 없습니다: {midi_file}")
                return True

            try:
                # 재생 볼륨 설정
                pygame.mixer.music.set_volume(0.7)
                pygame.mixer.music.load(midi_file)
                pygame.mixer.music.play()

                print("\n=== MIDI 파일 재생 중 ===")
                print("H: 재생 중지")
                print("P: 프로그램 종료")
                print("R: 처음으로 돌아가기")
                print("↑: 볼륨 증가")
                print("↓: 볼륨 감소")
                print("Space: 일시정지/재생")

                paused = False
                while pygame.mixer.music.get_busy() or paused:
                    key = cv2.waitKey(1) & 0xFF

                    # 재생 제어
                    if key == ord('h'):  # 중지
                        pygame.mixer.music.stop()
                        break
                    elif key == ord('p'):  # 프로그램 종료
                        pygame.mixer.music.stop()
                        return False
                    elif key == ord('r'):  # 처음으로
                        pygame.mixer.music.stop()
                        break
                    elif key == 32:  # Space - 일시정지/재생
                        if paused:
                            pygame.mixer.music.unpause()
                            print("재생 재개")
                        else:
                            pygame.mixer.music.pause()
                            print("일시정지")
                        paused = not paused

                    # 볼륨 제어
                    elif key == 82:  # Up arrow - 볼륨 증가
                        current_volume = pygame.mixer.music.get_volume()
                        pygame.mixer.music.set_volume(min(1.0, current_volume + 0.1))
                        print(f"볼륨: {pygame.mixer.music.get_volume():.1f}")
                    elif key == 84:  # Down arrow - 볼륨 감소
                        current_volume = pygame.mixer.music.get_volume()
                        pygame.mixer.music.set_volume(max(0.0, current_volume - 0.1))
                        print(f"볼륨: {pygame.mixer.music.get_volume():.1f}")

                    pygame.time.wait(100)

                return True

            except pygame.error as e:
                print(f"MIDI 재생 중 오류 발생: {e}")
                return True

        except Exception as e:
            print(f"MIDI 파일 처리 중 오류 발생: {e}")
            return True

        finally:
            try:
                pygame.mixer.quit()
                pygame.quit()
            except:
                pass

    def get_music_recommendations(self, gender, age_group, emotion):
        """음악 추천"""
        print("\n사용자 맞춤 음악을 검색하고 있습니다...")
        emotion_keywords = {
            "화남": "calming relaxing",
            "행복": "upbeat happy",
            "슬픔": "comforting healing",
            "중립": "ambient peaceful",
            "놀람": "soothing meditation",
            "공포": "peaceful calming",
            "혐오": "positive uplifting"
        }

        age_preferences = {
            "0-10": "children songs",
            "11-20": "pop kpop",
            "21-39": "indie alternative",
            "40-59": "classical jazz",
            "60+": "traditional healing"
        }

        try:
            base_query = f"{emotion_keywords[emotion]} music {age_preferences[age_group]}"
            request = self.youtube.search().list(
                part="snippet",
                maxResults=10,
                q=base_query,
                type="video"
            )

            response = request.execute()
            videos = response['items']
            weighted_choice = random.choices(
                videos,
                weights=[self.weights[emotion]] * len(videos),
                k=1
            )[0]

            return {
                'title': weighted_choice['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={weighted_choice['id']['videoId']}"
            }
        except Exception as e:
            print(f"YouTube 검색 중 오류 발생: {e}")
            return None

    def play_midi_file(self, midi_file):
        """MIDI 파일 재생"""
        try:
            # pygame 초기화
            mixer.init()
            mixer.music.load(midi_file)
            mixer.music.play()

            print("\n=== MIDI 파일 재생 중 ===")
            print("H: 재생 중지")
            print("P: 프로그램 종료")
            print("R: 처음으로 돌아가기")

            while mixer.music.get_busy():  # 음악이 재생 중일 때만 반복
                key = cv2.waitKey(1) & 0xFF
                if key == ord('h'):  # H key
                    mixer.music.stop()
                    break
                elif key == ord('p'):  # P key
                    mixer.music.stop()
                    return False
                elif key == ord('r'):  # R key
                    mixer.music.stop()
                    break
                time.sleep(0.1)

            mixer.quit()
            return True

        except Exception as e:
            print(f"MIDI 파일 재생 중 오류 발생: {e}")
            if mixer.get_init():
                mixer.quit()
            return True

    def play_youtube_video(self, video_url):
        """YouTube 비디오 재생"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--start-maximized')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            print(f"YouTube URL 열기: {video_url}")
            driver.get(video_url)

            wait = WebDriverWait(driver, 20)

            try:
                skip_button = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ytp-ad-skip-button"))
                )
                skip_button.click()
            except:
                pass

            try:
                play_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".ytp-large-play-button"))
                )
                driver.execute_script("arguments[0].click();", play_button)
            except:
                print("자동 재생 실패, 수동으로 재생해주세요.")

            # 키 설정 변경
            print("\n=== 재생 제어 ===")
            print("H: 재생 중지")
            print("P: 프로그램 종료")
            print("R: 음악 중지 후 처음으로 돌아가기")

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('h'):  # H key
                    break
                elif key == ord('p'):  # P key
                    return None  # 프로그램 종료 신호
                elif key == ord('r'):  # R key
                    break

            return driver

        except Exception as e:
            print(f"비디오 재생 중 오류 발생: {e}")
            if 'driver' in locals():
                driver.quit()
            return None

    def load_weights(self):
        """가중치 로드"""
        try:
            if os.path.exists(WEIGHT_FILE):
                with open(WEIGHT_FILE, 'r') as file:
                    return json.load(file)
            return self.default_weights.copy()
        except Exception as e:
            print(f"가중치 로드 중 오류 발생: {e}")
            return self.default_weights.copy()

    def save_weights(self):
        """가중치 저장"""
        try:
            with open(WEIGHT_FILE, 'w') as file:
                json.dump(self.weights, file, indent=4)
        except Exception as e:
            print(f"가중치 저장 중 오류 발생: {e}")

    def update_weights(self, emotion, liked=True):
        """가중치 업데이트"""
        try:
            if liked:
                self.weights[emotion] = round(self.weights[emotion] + 0.1, 2)
            else:
                self.weights[emotion] = max(0.1, round(self.weights[emotion] - 0.1, 2))
            self.save_weights()
        except Exception as e:
            print(f"가중치 업데이트 중 오류 발생: {e}")

    @staticmethod
    def classify_age(age):
        """나이 그룹 분류"""
        if age <= 10:
            return "0-10"
        elif age <= 20:
            return "11-20"
        elif age <= 39:
            return "21-39"
        elif age <= 59:
            return "40-59"
        else:
            return "60+"

    def play_youtube_video(self, video_url):
        """YouTube 비디오 재생"""
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            options = webdriver.ChromeOptions()
            driver = webdriver.Chrome(service=service, options=options)
            driver.get(video_url)

            play_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ytp-large-play-button.ytp-button"))
            )
            play_button.click()

            return driver
        except Exception as e:
            print(f"비디오 재생 중 오류 발생: {e}")
            if driver:
                driver.quit()
            return None

    def analyze_emotions(self):
        """감정 분석 실행"""
        # 웹캠에서 얼굴 인식
        frame = self.capture_image_until_face()
        if frame is None:
            return None

        try:
            # 얼굴 분석
            faces = self.detector.detect_faces(frame)
            face = faces[0]  # 이미 얼굴이 인식된 상태
            x, y, w, h = face['box']
            face_frame = frame[y:y + h, x:x + w]

            # DeepFace 분석 결과 출력 추가
            face_analysis = DeepFace.analyze(face_frame,
                                             actions=['age', 'gender'],
                                             enforce_detection=False,
                                             silent=True)  # 진행률 표시줄 숨기기

            print("\n=== 얼굴 분석 결과 ===")
            if isinstance(face_analysis, list):
                face_analysis = face_analysis[0]
            print(f"감지된 성별: {face_analysis['gender']}")
            print(f"추정 나잇대: {face_analysis['age']}")

        except Exception as e:
            print(f"얼굴 분석 중 오류 발생: {e}")
            return None

        # 음성 녹음 및 분석
        audio_file = self.record_audio_stream()
        if audio_file is None:
            return None

        try:
            emotion = self.analyze_audio_emotion(audio_file)
        except Exception as e:
            print(f"오디오 감정 분석 중 오류 발생: {e}")
            return None

        return {
            'gender': 'Male' if face_analysis['gender']['Man'] > face_analysis['gender']['Woman'] else 'Female',
            'age_group': self.classify_age(face_analysis['age']),
            'emotion': emotion
        }

    def analyze_audio_emotion(self, audio_path):
        """오디오 감정 분석"""
        print("\n음성을 분석하고 있습니다...")
        try:
            with torch.no_grad():
                # 오디오 처리
                waveform, sample_rate = torchaudio.load(audio_path)

                # 스테레오를 모노로 변환
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # 샘플레이트 변환
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)

                # 차원 조정
                waveform = waveform.squeeze(0)  # Remove extra dimension
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)

                # 장치 이동
                waveform = waveform.to(self.device)

                # 감정 분석
                audio_emotion = self.audio_model(waveform).logits

                # STT 변환
                client = speech.SpeechClient()
                with open(audio_path, "rb") as audio_file:
                    content = audio_file.read()

                response = client.recognize(
                    config=speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        language_code="ko-KR"
                    ),
                    audio=speech.RecognitionAudio(content=content)
                )

                transcript = ""
                for result in response.results:
                    transcript = result.alternatives[0].transcript

                # 텍스트 감정 분석
                text_input = self.tokenizer(transcript, padding=True, truncation=True, return_tensors="pt")
                text_input = {k: v.to(self.device) for k, v in text_input.items()}
                text_emotion = self.text_model(**text_input)

                # 결과 결합
                combined_emotion = 0.9 * text_emotion + 0.1 * audio_emotion
                emotion_idx = torch.argmax(torch.softmax(combined_emotion, dim=1)).item()

                emotions = ["화남", "행복", "슬픔", "중립", "놀람", "공포", "혐오"]
                return emotions[emotion_idx]

        except Exception as e:
            print(f"오디오 감정 분석 중 오류 발생: {e}")
            raise

    def create_music(self, emotion):
        """AI 작곡"""
        print(f"\n{emotion} 감정에 맞는 음악을 작곡하고 있습니다...")
        emotion_config = self.emotion_instruments[emotion.lower()]
        tempo = random.randint(*emotion_config["tempo"])

        midi = MIDIFile(len(emotion_config["instruments"]))

        for track, instrument in enumerate(emotion_config["instruments"]):
            midi.addTempo(track, 0, tempo)
            midi.addProgramChange(track, track, 0, instrument)

            # 기본 멜로디 생성
            base_note = 60
            for i in range(16):
                note = base_note + random.randint(-5, 5)
                midi.addNote(track, track, note, i, 1, random.randint(60, 100))

        filename = f"generated_{emotion}.mid"
        with open(filename, "wb") as output_file:
            midi.writeFile(output_file)
        return filename

    def get_music_recommendations(self, gender, age_group, emotion):
        """음악 추천"""
        print("\n사용자 맞춤 음악을 검색하고 있습니다...")
        emotion_keywords = {
            "화남": "calming relaxing",
            "행복": "upbeat happy",
            "슬픔": "comforting healing",
            "중립": "ambient peaceful",
            "놀람": "soothing meditation",
            "공포": "peaceful calming",
            "혐오": "positive uplifting"
        }

        age_preferences = {
            "0-10": "children songs",
            "11-20": "pop kpop",
            "21-39": "indie alternative",
            "40-59": "classical jazz",
            "60+": "traditional healing"
        }

        base_query = f"{emotion_keywords[emotion]} music {age_preferences[age_group]}"

        try:
            request = self.youtube.search().list(
                part="snippet",
                maxResults=10,
                q=base_query,
                type="video"
            )

            response = request.execute()
            videos = response['items']
            weighted_choice = random.choices(
                videos,
                weights=[self.weights[emotion]] * len(videos),
                k=1
            )[0]

            return {
                'title': weighted_choice['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={weighted_choice['id']['videoId']}"
            }
        except Exception as e:
            print(f"YouTube 검색 중 오류 발생: {e}")
            return None


def play_youtube_video(self, video_url):
    """YouTube 비디오 재생"""
    try:
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(video_url)

        play_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ytp-large-play-button.ytp-button"))
        )
        play_button.click()

        return driver
    except Exception as e:
        print(f"비디오 재생 중 오류 발생: {e}")
        return None


def load_weights(self):
    """가중치 로드"""
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, 'r') as file:
            return json.load(file)
    return self.default_weights.copy()

    def save_weights(self):
        """가중치 저장"""
        try:
            with open(WEIGHT_FILE, 'w') as file:
                json.dump(self.weights, file, indent=4)
        except Exception as e:
            print(f"가중치 저장 중 오류 발생: {e}")

    def update_weights(self, emotion, liked=True):
        """가중치 업데이트"""
        try:
            if liked:
                self.weights[emotion] = round(self.weights[emotion] + 0.1, 2)
            else:
                self.weights[emotion] = max(0.1, round(self.weights[emotion] - 0.1, 2))
            self.save_weights()
        except Exception as e:
            print(f"가중치 업데이트 중 오류 발생: {e}")

    @staticmethod
    def classify_age(age):
        """나이 그룹 분류"""
        if age <= 10:
            return "0-10"
        elif age <= 20:
            return "11-20"
        elif age <= 39:
            return "21-39"
        elif age <= 59:
            return "40-59"
        else:
            return "60+"


class WebcamWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_size = QSize(640, 480)
        self.setup_ui()
        self.setup_camera()

    def stop_camera(self):
        if hasattr(self, 'capture'):
            self.timer.stop()
            self.capture.release()
            self.capture = None

    def start_camera(self):
        if not hasattr(self, 'capture') or self.capture is None:
            self.setup_camera()

    def setup_ui(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.capture_button = QPushButton("감정 분석 시작")
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.capture_button)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    def setup_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.video_size, Qt.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(scaled_image))


class ResultWidget(QWidget):
    def __init__(self, emotion_data, parent=None):
        super().__init__(parent)
        self.emotion_data = emotion_data
        self.setup_ui()
        self.is_playing = False

    def setup_ui(self):
        layout = QVBoxLayout()

        # 타이틀
        title = QLabel("오늘의 감정 분석 결과")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                margin: 20px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 감정 결과
        self.emotion_label = QLabel()
        self.emotion_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                margin: 20px;
            }
        """)
        self.emotion_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.emotion_label)

        # 감정 설명
        self.description_label = QLabel()
        self.description_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                margin: 20px;
            }
        """)
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        # 분석 결과 상세
        self.details_label = QLabel()
        self.details_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                margin: 20px;
            }
        """)
        self.details_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.details_label)

        # 음악 선택 버튼
        music_group = QGroupBox("음악 선택")
        music_layout = QHBoxLayout()

        self.ai_music_button = QPushButton("AI 작곡 음악")
        self.youtube_music_button = QPushButton("YouTube 음악")

        for button in [self.ai_music_button, self.youtube_music_button]:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 14px;
                    min-width: 150px;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
            music_layout.addWidget(button)

        music_group.setLayout(music_layout)
        layout.addWidget(music_group)

        # YouTube 플레이어 영역
        self.youtube_widget = QWidget()
        self.youtube_widget.hide()
        layout.addWidget(self.youtube_widget)

        # 하단 버튼
        button_layout = QHBoxLayout()
        self.new_button = QPushButton("새로운 일기 쓰기")
        self.new_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(self.new_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def update_result(self, gender, age_group, emotion):
        emoji = self.emotion_data[emotion]['emoji']
        description = self.emotion_data[emotion]['description']

        self.emotion_label.setText(f"{emoji}\n오늘의 감정: {emotion}")
        self.description_label.setText(description)
        self.details_label.setText(f"성별: {gender}\n나이대: {age_group}\n감정: {emotion}")


class EmotionAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.emotion_data = {
            '화남': {'emoji': '😠', 'description': '화난 감정이 감지되었습니다. 잠시 심호흡을 하고 마음을 진정시켜보는 건 어떨까요?'},
            '행복': {'emoji': '😊', 'description': '행복한 감정이 감지되었습니다. 이 기분 좋은 순간을 기억해두세요!'},
            '슬픔': {'emoji': '😢', 'description': '슬픈 감정이 감지되었습니다. 때로는 슬픔을 느끼는 것도 자연스러운 일이에요.'},
            '중립': {'emoji': '😐', 'description': '평온한 감정 상태입니다. 차분히 하루를 되돌아보세요.'},
            '놀람': {'emoji': '😮', 'description': '놀란 감정이 감지되었습니다. 예상치 못한 일이 있었나요?'},
            '공포': {'emoji': '😨', 'description': '두려운 감정이 감지되었습니다. 혼자 힘들어하지 마세요.'},
            '혐오': {'emoji': '😫', 'description': '불쾌한 감정이 감지되었습니다. 기분 전환이 필요해 보여요.'}
        }

        self.analyzer = EmotionAnalyzer()
        self.current_music_file = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('감정 일기장')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)
        # 스택 위젯 설정
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 웹캠 위젯
        self.webcam = WebcamWidget()
        self.webcam.capture_button.clicked.connect(self.start_analysis)
        self.stack.addWidget(self.webcam)

        # 결과 위젯
        self.result = ResultWidget(self.emotion_data)
        self.result.new_button.clicked.connect(self.new_analysis)
        self.result.ai_music_button.clicked.connect(self.play_ai_music)
        self.result.youtube_music_button.clicked.connect(self.play_youtube_music)
        self.stack.addWidget(self.result)

        self.setMinimumSize(800, 600)
        self.center_window()

    def center_window(self):
        frame = self.frameGeometry()
        center = QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center)
        self.move(frame.topLeft())

    def start_analysis(self):
        self.webcam.stop_camera()  # 웹캠 중지
        results = self.analyzer.analyze_emotions()

        if results:
            gender = '남성' if results['gender'] == 'Male' else '여성'
            self.result.update_result(gender, results['age_group'], results['emotion'])
            self.stack.setCurrentWidget(self.result)
        else:
            self.webcam.start_camera()  # 분석 실패시 웹캠 재시작
            QMessageBox.warning(self, "오류", "분석에 실패했습니다. 다시 시도해주세요.")

    def new_analysis(self):
        self.webcam.start_camera()  # 웹캠 재시작
        self.stack.setCurrentWidget(self.webcam)

    def play_ai_music(self):
        try:
            emotion = self.result.emotion_label.text().split(': ')[1]
            self.current_music_file = self.analyzer.create_music(emotion)
            print(f"음악 파일이 생성되었습니다: {self.current_music_file}")

            # 파일 존재 확인
            if not os.path.exists(self.current_music_file):
                raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다: {self.current_music_file}")

            print("음악 재생 시도...")
            # 시스템 기본 프로그램으로 실행
            if sys.platform == 'win32':
                os.system(f'start wmplayer "{self.current_music_file}"')
            else:
                os.system(f'open "{self.current_music_file}"')

            # 버튼 상태 업데이트
            self.result.ai_music_button.setEnabled(False)
            self.result.youtube_music_button.setEnabled(False)
            self.result.is_playing = True

        except Exception as e:
            print(f"음악 재생 중 오류 발생: {e}")
            QMessageBox.warning(self, "오류", f"음악 재생 중 오류가 발생했습니다: {str(e)}")

            # 버튼 상태 복구
            self.result.ai_music_button.setEnabled(True)
            self.result.youtube_music_button.setEnabled(True)
            self.result.is_playing = False

        except Exception as e:
            print(f"음악 재생 중 오류 발생: {e}")
            QMessageBox.warning(self, "오류", f"음악 재생 중 오류가 발생했습니다: {str(e)}")

            # 버튼 상태 복구
            self.result.stop_button.setEnabled(False)
            self.result.ai_music_button.setEnabled(True)
            self.result.youtube_music_button.setEnabled(True)
            self.result.is_playing = False

    def stop_music(self):
        try:
            if hasattr(self, 'current_player'):
                if self.current_player == 'winsound':
                    import winsound
                    winsound.PlaySound(None, winsound.SND_PURGE)
                elif self.current_player == 'media_player':
                    if sys.platform == 'win32':
                        os.system('taskkill /F /IM wmplayer.exe')
        except Exception as e:
            print(f"음악 중지 중 오류 발생: {e}")

        # 버튼 상태 업데이트
        self.result.stop_button.setEnabled(False)
        self.result.ai_music_button.setEnabled(True)
        self.result.youtube_music_button.setEnabled(True)
        self.result.is_playing = False

    def play_youtube_music(self):
        current_emotion = self.result.emotion_label.text().split(': ')[1]
        recommendation = self.analyzer.get_music_recommendations(
            self.result.details_label.text().split('\n')[0].split(': ')[1],
            self.result.details_label.text().split('\n')[1].split(': ')[1],
            current_emotion
        )
        if recommendation:
            driver = self.analyzer.play_youtube_video(recommendation['url'])
            if driver:
                # 피드백 대화상자 표시
                feedback = QMessageBox.question(
                    self,
                    "음악 피드백",
                    "음악이 마음에 드셨나요?",
                    QMessageBox.Yes | QMessageBox.No
                )
                # 피드백 저장
                self.analyzer.update_weights(
                    current_emotion,
                    feedback == QMessageBox.Yes
                )
                driver.quit()

# 메인 함수
def main():
    """메인 실행 함수"""
    app = QApplication(sys.argv)
    window = EmotionAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()