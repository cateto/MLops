# MLflow Tracking

### 각 Running 에서 기록되는 정보

![Untitled](MLflow%20Tracking/Untitled.png)

- Source : 모델 파일 명
- Start Time & End Time : 시작 및 종료시간
- Parameters (key -value 구조)
- Metrics
- Artifacts (모델 파일, 데이터 파일)

### [ 예시 : IMDB 영화 감성 분류 모델 훈련 ]

# 시나리오 1 : localhost에서 Local File System에 Artifact 저장

### 아키텍처 이미지

![Untitled](MLflow%20Tracking/Untitled%201.png)

## Version 1 ) Local - manually

```python
import tensorflow
import os
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow

#이미 훈련, 테스트 데이터가 50:50로 구분되어 제공됨.
# 영화 리뷰는 X_train에, 감성 정보는 y_train에 저장된다.
# 테스트용 리뷰는 X_test에, 테스트용 리뷰의 감성 정보는 y_test에 저장된다.
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 10000)
# 상위 10000건의 단어들만 사용.

if __name__ == '__main__':
    **mlflow.set_experiment('classfication')**
    env = 'local'
    **mlflow.log_param('env', env)**

    print('1. Load Data')
    vocab_size = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    print('훈련용 리뷰 개수 : {}'.format(len(x_train)))
    print('테스트용 리뷰 개수 : {}'.format(len(x_test)))
    num_classes = max(y_train) + 1
    print('카테고리 : {}'.format(num_classes))
    
    print('2. Preprocessing')
    max_len = 500
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    print('3. Build Model')
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(GRU(128))
    model.add(Dense(1, activation='sigmoid'))

    print('4. Model Train')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1,save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    print("\n 테스트 정확도 : %.4f"%(model.evaluate(x_test, y_test)[1]))

    **mlflow.log_metric('accuracy', model.evaluate(x_test, y_test)[1])**

    # MLflow Tracking (parameter)
    import random
    random_no = random.randrange(0, len(x_train))

    **mlflow.log_param("train", 'from tensorflow.keras.datasets.imdb')
    mlflow.log_param("train num", len(x_train))
    mlflow.log_param("class num", num_classes)
    mlflow.log_param("class", {0:'negative', 1:'positive'})
    mlflow.log_param("train example", x_train[random_no])
    mlflow.log_param("train text max length", max([len(x) for x in x_train]))
    mlflow.log_param("train text average length", sum([len(x) for x in x_train])/len(x_train))

    mlflow.tensorflow.log_model(model, "model", pip_requirements=[f"tensorflow=={tensorflow.__version__}"])**
```

## Version 2 ) Local - Autolog

<aside>
💡 tf2도 적용한 코드가 있다면 포함!

</aside>

```python
import os
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
from pprint import pprint
from mlflow.tracking.client import MlflowClient

#이미 훈련, 테스트 데이터가 50:50로 구분되어 제공됨.
# 영화 리뷰는 X_train에, 감성 정보는 y_train에 저장된다.
# 테스트용 리뷰는 X_test에, 테스트용 리뷰의 감성 정보는 y_test에 저장된다.
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 10000)
# 상위 10000건의 단어들만 사용.

def fetch_logged_data(run_id):
    client = MlflowClient()
    return client.get_run(run_id).to_dictionary()['data']

if __name__ == '__main__':
    **mlflow.tensorflow.autolog()**
    **mlflow.set_experiment('classfication-Autolog')**

    print('1. Load Data')
    vocab_size = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    print('훈련용 리뷰 개수 : {}'.format(len(x_train)))
    print('테스트용 리뷰 개수 : {}'.format(len(x_test)))
    num_classes = max(y_train) + 1
    print('카테고리 : {}'.format(num_classes))
    
    print('2. Preprocessing')
    max_len = 500
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    if(mlflow.active_run):
        mlflow.end_run()

    **with mlflow.start_run() as run:**
        print('3. Build Model')
        model = Sequential()
        model.add(Embedding(vocab_size, 100))
        model.add(GRU(128))
        model.add(Dense(1, activation='sigmoid'))

        print('4. Model Train')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1,save_best_only=True)

        print("Logged data and model in run {}".format(run.info.run_id))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
        print("\n 테스트 정확도 : %.4f"%(model.evaluate(x_test, y_test)[1]))

         # show logged data
        for key, data in fetch_logged_data(run.info.run_id).items():
            print("\n---------- logged {} - ---------".format(key))
            pprint(data)
```

# 시나리오 2 : Tracking Server DB와 Local File System에 Artifact 저장

### 아키텍처 이미지

![Untitled](MLflow%20Tracking/Untitled%202.png)

# 시나리오 3 : Tracking Server DB와 SFTP를 통한 Remote File System에 Artifact 저장

### 아키텍처 이미지

![Untitled](MLflow%20Tracking/Untitled%203.png)

## Version 1 ) Server - manually

<aside>
💡 MLflow 프로젝트를 실행시키기 위해서 backend server와 연결해주는 작업이 필요하다. 이는 간단하게 환경변수 설정으로 해결할 수 있다.

```bash
$ export MLFLOW_TRACKING_URI="{서버의 URI}"
```

</aside>

```python
import tensorflow
import os
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow

#이미 훈련, 테스트 데이터가 50:50로 구분되어 제공됨.
# 영화 리뷰는 X_train에, 감성 정보는 y_train에 저장된다.
# 테스트용 리뷰는 X_test에, 테스트용 리뷰의 감성 정보는 y_test에 저장된다.
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 10000)
# 상위 10000건의 단어들만 사용.

**artifact_uri = 'sftp:///mlops@192.168.1.70/home/mlops/mlflow/mlruns' # 환경변수로 지정 가능한지 테스트**

if __name__ == '__main__':
    # uncomment line below when you want to set manually server to track (if you set ENV you don't have to uncomment line below )
    # mlflow.set_tracking_uri({SERVER URI})
		experiment_name = '**classfication**'
    if(mlflow.get_experiment_by_name(experiment_name)):
        **mlflow.set_registry_uri(artifact_uri)**
    else:
        **mlflow.create_experiment(name=experiment_name, artifact_location=artifact_uri)**
    **mlflow.set_experiment(experiment_name=experiment_name)**
    env = ''
    if(os.environ['MLFLOW_TRACKING_URI']):
        env = os.environ['MLFLOW_TRACKING_URI']
    # elif({SERVER URI}):
    #     env = {SERVER URI}
    else:
        env = 'local'
    **mlflow.log_param('env', env)**

    print('1. Load Data')
    vocab_size = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    print('훈련용 리뷰 개수 : {}'.format(len(x_train)))
    print('테스트용 리뷰 개수 : {}'.format(len(x_test)))
    num_classes = max(y_train) + 1
    print('카테고리 : {}'.format(num_classes))
    
    print('2. Preprocessing')
    max_len = 500
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    print('3. Build Model')
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(GRU(128))
    model.add(Dense(1, activation='sigmoid'))

    print('4. Model Train')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1,save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    print("\n 테스트 정확도 : %.4f"%(model.evaluate(x_test, y_test)[1]))

    **mlflow.log_metric('accuracy', model.evaluate(x_test, y_test)[1])**

    # MLflow Tracking (parameter)
    import random
    random_no = random.randrange(0, len(x_train))

    **mlflow.log_param("train", 'from tensorflow.keras.datasets.imdb')
    mlflow.log_param("train num", len(x_train))
    mlflow.log_param("class num", num_classes)
    mlflow.log_param("class", {0:'negative', 1:'positive'})
    mlflow.log_param("train example", x_train[random_no])
    mlflow.log_param("train text max length", max([len(x) for x in x_train]))
    mlflow.log_param("train text average length", sum([len(x) for x in x_train])/len(x_train))

    mlflow.tensorflow.log_model(model, "model", pip_requirements=[f"tensorflow=={tensorflow.__version__}"])**
```

## Version 2 ) Server - Autolog

<aside>
💡 MLflow 프로젝트를 실행시키기 위해서 backend server와 연결해주는 작업이 필요하다. 이는 간단하게 환경변수 설정으로 해결할 수 있다.

```bash
$ export MLFLOW_TRACKING_URI="{서버의 URI}"
```

</aside>

```python
import os
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
from pprint import pprint
from mlflow.tracking.client import MlflowClient

#이미 훈련, 테스트 데이터가 50:50로 구분되어 제공됨.
# 영화 리뷰는 X_train에, 감성 정보는 y_train에 저장된다.
# 테스트용 리뷰는 X_test에, 테스트용 리뷰의 감성 정보는 y_test에 저장된다.
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 10000)
# 상위 10000건의 단어들만 사용.

def fetch_logged_data(run_id):
     # uncomment line below when you want to set manually server to track (if you set ENV you don't have to uncomment line below )
    # client = MlflowClient({SERVER URI})
    client = MlflowClient()
    return client.get_run(run_id).to_dictionary()['data']

if __name__ == '__main__':
		**mlflow.tensorflow.autolog()**
    
		# uncomment line below when you want to set manually server to track (if you set ENV you don't have to uncomment line below )
    # mlflow.set_tracking_uri({SERVER URI})
		
		experiment_name = '**classfication-auto**'
    if(mlflow.get_experiment_by_name(experiment_name)):
        pass
    else:
        **mlflow.create_experiment(name=experiment_name)**
    **mlflow.set_experiment(experiment_name=experiment_name)**
    env = ''
    if(os.environ['MLFLOW_TRACKING_URI']):
        env = os.environ['MLFLOW_TRACKING_URI']
    # elif({SERVER URI}):
    #     env = {SERVER URI}
    else:
        env = 'local'
    **mlflow.log_param('env', env)**
    
    
****
    print('1. Load Data')
    vocab_size = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    print('훈련용 리뷰 개수 : {}'.format(len(x_train)))
    print('테스트용 리뷰 개수 : {}'.format(len(x_test)))
    num_classes = max(y_train) + 1
    print('카테고리 : {}'.format(num_classes))
    
    print('2. Preprocessing')
    max_len = 500
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    if(mlflow.active_run):
        mlflow.end_run()

    **with mlflow.start_run() as run:**
        print('3. Build Model')
        model = Sequential()
        model.add(Embedding(vocab_size, 100))
        model.add(GRU(128))
        model.add(Dense(1, activation='sigmoid'))

        print('4. Model Train')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1,save_best_only=True)

        print("Logged data and model in run {}".format(run.info.run_id))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
        print("\n 테스트 정확도 : %.4f"%(model.evaluate(x_test, y_test)[1]))

         # show logged data
        for key, data in fetch_logged_data(run.info.run_id).items():
            print("\n---------- logged {} - ---------".format(key))
            pprint(data)
```

# Tracking Server

### **시나리오 2 : Tracking Server 구축 (**🖱️)

<aside>
💡 여러명의 모델러들이 모델 개발을 할 때 로그를 중앙에서 관리할 서버가 필요하다. 
MLflow는 Tracking 역할을 위한 서버를 제공한다. 이를 Tracking Server라고 한다. 
Local에서 작업할 때는 `./mlruns` 에 바로 로그와 모델이 저장되었다면 이제는 백엔드 서버를 통해서 저장할 수 있다.

</aside>

아래와 같은 명령어를 통해 간단하게 Tracking Server를 띄울 수 있다.

```bash
$ mkdir tracking-server
$ cd tracking-server

$ mlflow server -h 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root $(pwd)/artifacts

```

여기서 `--backend-store-uri` 와 `--default-artifact-root` 는 밑에서 다시 자세히 확인해보도록 하자.

이제 다시 MLflow의 프로젝트를 실행시켜보자. 

(위의 예시 중 server - blahblah를 실행시키면 됨.) 
프로젝트 실행 후에 아까 생성한 `tracking-server` 디렉토리로 가면 다음과 같은 실행 결과를 볼 수 있다.

```bash
.
├── artifacts
│   ├── 0
│   │   ├── 2df9ce14a7ec4431904f6e8292c08d67
│   │   │   └── artifacts
│   │   │       ├── model
│   │   │       │   ├── MLmodel
│   │   │       │   ├── conda.yaml
│   │   │       │   ├── requirements.txt
│   │   │       │   └── tfmodel
│   │   │       │       ├── saved_model.pb
│   │   │       │       └── variables
│   │   │       │           ├── variables.data-00000-of-00001
│   │   │       │           └── variables.index
│   │   │       └── tensorboard_logs
│   │   └──         └── events.out.tfevents.1644278684.RSN-DL
└── mlflow.db
```

`artifacts` : Tracking server를 실행시켰을 때 model 파일, 학습 dataset 파일 등이 저장됨.

`mlflow.db` : sqlite DB

### 시나리오 3 : Tracking Server 구축 및 트러블 슈팅 (🖱️)

<aside>
💡 여러명의 모델러들이 모델 개발을 할 때 로그를 중앙에서 관리할 서버가 필요하다. 
MLflow는 Tracking 역할을 위한 서버를 제공한다. 이를 Tracking Server라고 한다. 
Local에서 작업할 때는 `./mlruns` 에 바로 로그와 모델이 저장되었다면 이제는 백엔드 서버를 통해서 저장할 수 있다.

</aside>

[Setup MLflow in Production](https://medium.com/@gyani91/setup-mlflow-in-production-d72aecde7fef)

## sftp로 모델 아티팩트 관리하는 server로 간단하게 구축하기!

명령어를 통해 Tracking Server를 띄울 수 있지만, 자주 사용하는 명령어이므로 service에 등록해두자.

```bash
mlflow server --backend-store-uri sqlite:///home/mlops/mlflow/mlflow.db --default-artifact-root sftp:///mlops@192.168.1.70:/home/mlops/mlflow/mlruns
```

1. ssh 연동 (비밀번호 없이 sftp 접속할 수 있도록 함)
2. 공개키 등록 /user/.ssh는 물론 /root/.ssh 에도 등록해야함. 만약 tracking server라고 해도 거기서 model code가 있다면 그 키도 등록해줘야 함. (자기자신이라고하지만 sftp 프로토콜을 통해서 접속해야하기때문임 싫으면 알아서 그 디렉토리를 지정하던가!)
3. /root/.ssh에 등록하는 이유는 dashboard에서 모델 artifacts에 접근하기 위함임
4. `sftp://` url parsing이 안된다면?
    - 자세히
      
        일단은 set_tracking_uri만 할게아니라 registry_uri도 설정해줘야함. 환경변수에서 어떻게 불러오는지 추가 확인이 필요할듯.
        
        artifact_uri = 'sftp://mlops@192.168.1.70:22/home/mlops/mlflow/mlruns’
        
        이 형태로 써야함. 포트도 써줘야했음ㅎㅎㅎㅎㅎㅎ,,,,,,
        
        일단 설정에서 미흡했던 점은 ssh-keygen 에서 헷갈리는 점이 많았음. 
        
        (리눅스 계정의 공개키 등록 및 개인키 발급, local이라 당연히 ssh인증없이 sftp 접속 가능한 줄알았지만 사실은 우회해서 들어오므로 ssh 인증이 필요했음, home 디렉토리의 `~` 가 인식되지 않는 문제, 디렉토리의 권한 문제 등. 아직 해결되지 않은 문제는 mlflow ui 대시보드가 root 권한으로 sftp 인식하는지 mlops 권한으로 접근하는지 좀더 파봐야함)
        
        그리고 amazon s3를 사용하라는 내용이 대부분이었으며
        
        sftprepository 를 연동하는것에 대한 내용이 아직 공식문서에 없고 정말 극소수의 사례뿐임.
        
        겨우 github에서 소스 검색해서 찾아서 연동했네 고맙네 지구촌이여