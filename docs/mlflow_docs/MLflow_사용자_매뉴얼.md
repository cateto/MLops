# MLflow 사용자 매뉴얼

## MLflow 클라이언트 환경

1. python 버전을 확인합니다.

```bash
python --version
Python 3.7.0
```

2. 다음을 실행하여 MLflow를 설치합니다.

```bash
$ pip install mlflow
$ mlflow --version
mlflow, version 1.23.1
```

## 간단한 Quick Start

1. mlflow의 공식 github 코드를  로컬 저장소에 clone합니다.

```bash
git clone https://github.com/mlflow/mlflow
```

2. `examples/quickstart` 경로를 확인합니다.

```bash
$ ls -al
total 20
drwxrwxr-x  4 rsn-dl rsn-dl 4096  2월  4 17:30 .
drwxrwxr-x 35 rsn-dl rsn-dl 4096  2월  4 17:20 ..
-rw-rw-r--  1 rsn-dl rsn-dl  508  2월  7 11:09 mlflow_tracking.py
drwxrwxr-x  6 rsn-dl rsn-dl 4096  2월  7 14:12 mlruns
drwxrwxr-x  2 rsn-dl rsn-dl 4096  2월  4 17:30 outputs
```

3. `mlflow_tracking.py` 를 파이썬으로 실행합니다.

```bash
$ python mlflow_tracking.py
Running mlflow_tracking.py
```

4. 실행 후 다음과 같은 결과를 확인합니다.

```bash
$ tree .
.
├── mlflow_tracking.py
├── mlruns
│   └── 0
│       ├── 5eb4904dea6a4036988eea86c36a5d0d
│       │   ├── artifacts
│       │   │   └── test.txt
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   └── foo
│       │   ├── params
│       │   │   └── param1
│       │   └── tags
│       │       ├── mlflow.source.git.commit
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       └── mlflow.user
│       └── meta.yaml
└── outputs
    └── test.txt
```

5. `mlflow ui` 명령어로 해당 디렉토리(로컬)에 저장된 대시보드 웹서버를 확인할 수 있습니다.

- 기본 포트 설정값은 5000번
- `mlflow ui -h 0.0.0.0 -p 2000` 와 같은 명령어로 ***포트***와 ***인바운드 규칙***를 변경할 수 있다.



### 기본 구성 이해하기

```bash
cd mlflow/examples/quickstart
```

1. Tracking API 사용 예시
   - 모델 학습 과정에서 parameter, metrics, artifact를 기록하고 버전 관리를 하며, 모델 학습 실행 기록을 볼 수 있다.

```python
## mlflow_tracking.py

import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    log_param("param1", randint(0, 100))

    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")
```

- [log_metric](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric)
- [log_param](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param)
- [log_artifacts](<https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifacts>)

위와 같은 3가지 함수를 가져와 Tracking하는데, 해당 링크를 클릭하여 자세한 함수의 설명을 확인할수 있다.

```python
python mlflow_tracking.py
```

실행 후에는 `mlruns`와 `outputs` 디렉토리가 생겨있다.

```bash
$ tree .

├── mlflow_tracking.py
├── mlruns
│   └── 0
│       ├── fd97b204ecb149b8bf5bb41674d6287c
│       │   ├── artifacts
│       │   │   └── test.txt
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   └── foo
│       │   ├── params
│       │   │   └── param1
│       │   └── tags
│       │       ├── mlflow.source.git.commit
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       └── mlflow.user
│       └── meta.yaml
└── outputs
    └── test.txt
```

- `metrics`
- `params`
- `artifacts`

위에서 `log_blahblah` 함수로 기록했던 값이 파일로 기록되어 있다. 특히 `metrics` 의 경우는 timestamp가 같이 기록되어 있다.

```bash
7b204ecb149b8bf5bb41674d6287c/metrics$ vi foo
1643963402394 0.6164196006416202 0
1643963402394 1.1171855651628642 0
1643963402395 2.4314951719135935 0
```

폴더 상단에서 (현재와 동일한 경우 ./mlflow/examples/quickstart) `mlflow ui` 명령어로 대시보드용 웹 서버를 띄울 수 있다. 다만, 서버 컴퓨터에서 실행시키고 (사설망) 접속하고자 하는 경우 `mlflow ui -h 0.0.0.0` 명령어로 실행하면 문제가 해결된다.



## 현재 사용가능한 MLFLOW_TRACKING_URI

2022년 2월 17일 기준 `[http://192.168.1.70:5000](http://192.168.1.70:5000)` 입니다.

- **방법 1 ) shell에서 환경변수로 설정 (권장)**

```bash
$ echo $MLFLOW_TRACKING_URI

$ export MLFLOW_TRACKING_URI=[http://192.168.1.70:5000](http://192.168.1.70:5000)

$ echo $MLFLOW_TRACKING_URI
http://192.168.1.70:5000
```

- 방법 2 ) 코드실행 시 설정

```python
server ='http://192.168.1.70:5000'
mlflow.set_tracking_uri(server)
```