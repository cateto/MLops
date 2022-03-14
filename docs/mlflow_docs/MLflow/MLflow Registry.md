# MLflow Registry

### MLflow 모델 레지스트리 기능을 사용하면 중앙 repository에서 ML 모델의 수명 주기를 컨트롤  할 수 있습니다. 또한 이는 배포할 모델의 품질을 보장하고, 모델의 버전 별 검색을 가능하게 합니다.

- 훈련 및 배포된 ML 모델을 등록, 추적, 버전관리를 할 수 있습니다.
- 모델의 메타데이터 및 런타임 환경을 저장합니다.
- 모델에 대한 metric을 추적합니다.
- 모델 프로덕션 프로세스를 관리합니다. (ex : staging에서 production으로 상태 변경)

### Step

1. mlflow 모델 세부 정보 페이지에서 Artifact 섹션에 기록된 mlflow 모델을 선택합니다.
2. `Register Model` 버튼을 클릭합니다.
   
    ![Untitled](MLflow%20Registry/Untitled.png)
    
3. `Model Name` 필드에서 새 모델을 추가하는 경우 모델을 식별하는 고유한 이름을 지정합니다. 혹은 기존 모델에 새 버전을 등록하는 경우에는 `Model` 의 드롭다운에서 기존 모델 이름을 선택합니다.`
   
    ![Untitled](MLflow%20Registry/Untitled%201.png)
    
4. `Models` 페이지에서 등록된 모델의 속성을 확인합니다.
   
    ![Untitled](MLflow%20Registry/Untitled%202.png)
    
5. 세부 페이지의 Artifacts 섹션으로 이동하여 모델을 클릭한다음, 오른쪽 상단의 모델 버전을 클릭하여 방금 생성한 모델의 버전을 확인합니다.
   
    ![Untitled](MLflow%20Registry/Untitled%203.png)
    
6. `Models` 버전 세부 정보 페이지에는 모델 버전의 디테일과, 모델 버전의 단계를 볼 수 있습니다.  `Stage` 드롭 다운을 클릭하여 모델 버전을 다른 버전으로 전환할 수도 있습니다.
   
    ![Untitled](MLflow%20Registry/Untitled%204.png)