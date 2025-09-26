// Jenkinsfile
def PROJECT_NAME = "autotrade-binance-app"
def Namespace    = "auto-coin"
def gitUrl       = "https://github.com/dslee1371/autotrade-binance"
def imgRegistry  = "172.10.30.11:5000"
def gitOpsUrl    = "https://github.com/dslee1371/gitops"
def opsBranch    = "main"
def GIT_TAG_MESSAGE

pipeline {
  agent {
    kubernetes {
      yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: kaniko
    image: gcr.io/kaniko-project/executor:debug
    command: ["/busybox/cat"]
    tty: true
    volumeMounts:
    - name: kaniko-secret
      mountPath: /kaniko/.docker
      readOnly: true
  - name: git
    image: alpine/git:latest
    command: ["cat"]
    tty: true
  volumes:
  - name: kaniko-secret
    secret:
      secretName: kaniko-secret
"""
    }
  }
  parameters {
        booleanParam(name: 'UPDATE_GITOPS', defaultValue: true, description: 'Update GitOps repository')
  }
  environment {
    DOCKER_CONFIG = '/kaniko/.docker'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout([
          $class: 'GitSCM',
          branches: [[name: "${params.TAG}"]], // 태그 체크아웃 시 refs/tags/${params.TAG} 사용 가능
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'CloneOption', depth: 1, shallow: true]],
          gitTool: 'Default',
          submoduleCfg: [],
          userRemoteConfigs: [[
            url: "${gitUrl}",
            credentialsId: 'github-auth'
          ]]
        ])

        script {
          GIT_TAG_MESSAGE = sh(
            script: 'git log -1 --pretty=format:"%h - %s"',
            returnStdout: true
          ).trim()
          echo "Building commit: ${GIT_TAG_MESSAGE}"
        }
      }
    }

    stage('Build and Push Image') {
      steps {
        container('kaniko') {
          script {
            def imageTag  = "${imgRegistry}/${Namespace}/${PROJECT_NAME}:${params.TAG}"
            def latestTag = "${imgRegistry}/${Namespace}/${PROJECT_NAME}:latest"

            withEnv(["IMAGE_TAG=${imageTag}", "LATEST_TAG=${latestTag}", "WORKSPACE_DIR=${env.WORKSPACE}"]) {
              sh '''
/kaniko/executor \
  --context=$WORKSPACE_DIR \
  --dockerfile=$WORKSPACE_DIR/Dockerfile \
  --destination=$IMAGE_TAG \
  --destination=$LATEST_TAG \
  --insecure \
  --skip-tls-verify \
  --cache=true \
  --cache-ttl=24h
'''
            }

            echo "Successfully built and pushed:"
            echo "- ${imageTag}"
            echo "- ${latestTag}"
          }
        }
      }
    }

    stage('Update GitOps Repository') {
      when { expression { params.UPDATE_GITOPS } }
      steps {
        container('git') {
          script {
            withCredentials([usernamePassword(
              credentialsId: 'github-auth',
              usernameVariable: 'GIT_USERNAME',
              passwordVariable: 'GIT_PASSWORD'
            )]) {
              def escUser = env.GIT_USERNAME.replaceAll('@','%40')

              // GitOps 앱 디렉터리 (필요 시 수정)
              def appDir = 'autotrade-binance'

              withEnv([
                "TAG=${params.TAG}",
                "IMG_REGISTRY=${imgRegistry}",
                "NAMESPACE=${Namespace}",
                "PROJECT_NAME=${PROJECT_NAME}",
                "OPS_BRANCH=${opsBranch}",
                "ESC_USER=${escUser}",
                "GIT_TAG_MESSAGE=${GIT_TAG_MESSAGE}",
                "APP_DIR=${appDir}"
              ]) {
                sh '''
set -euo pipefail

git config --global user.email "dslee1371@gmail.com"
git config --global user.name  "dslee"

# Clone GitOps repository
git clone https://github.com/dslee1371/gitops.git gitops-repo
cd gitops-repo
git checkout -B "$OPS_BRANCH" || true

ran_any=false

# 1) kustomization.yaml: newTag 값 교체
if [ -f "$APP_DIR/kustomization.yaml" ]; then
  sed -i -E 's|(^[[:space:]]*newTag:[[:space:]]*).*$|\\1'"$TAG"'|' "$APP_DIR/kustomization.yaml"
  echo "Updated $APP_DIR/kustomization.yaml to tag $TAG"
  ran_any=true
fi

# 2) deployment.yaml: image 태그 + 라벨 버전(version / app.kubernetes.io/version) 동시 교체
if [ -f "$APP_DIR/deployment.yaml" ]; then
  # 2-1) image 태그: 대상 프로젝트 이미지만, 다이제스트(@sha256)는 제외
  sed -i -E '/@sha256/! s|(image:[[:space:]]*[^[:space:]"]*/'"$PROJECT_NAME"'):[^[:space:]"#]+|\\1:'"$TAG"'|' "$APP_DIR/deployment.yaml"

  # 2-2) metadata.labels.version (따옴표 없이 설정)
  sed -i -E 's|(^[[:space:]]*version:[[:space:]]*).*$|\\1'"$TAG"'|' "$APP_DIR/deployment.yaml"

  # 2-3) metadata.labels."app.kubernetes.io/version" (항상 따옴표 유지)
  sed -i -E 's|(^[[:space:]]*app\\.[Kk]ubernetes\\.[Ii]o/version:[[:space:]]*).*$|\\1"'"$TAG"'"|' "$APP_DIR/deployment.yaml"

  echo "Updated $APP_DIR/deployment.yaml: image tag + labels(version, app.kubernetes.io/version) -> $TAG"
  ran_any=true
fi

# 3) 둘 다 없으면 경고
if [ "$ran_any" = false ]; then
  echo "Warning: No kustomization.yaml or deployment.yaml found under $APP_DIR"
  echo "Please update your GitOps repository structure"
fi

# Commit & push
git add -A
if git diff --staged --quiet; then
  echo "No changes to commit"
else
  git commit -m "Update $PROJECT_NAME image & label versions to $TAG" \
              -m "Build info: $GIT_TAG_MESSAGE" \
              -m "Jenkins Build: $BUILD_NUMBER"
  git push "https://$ESC_USER:$GIT_PASSWORD@github.com/dslee1371/gitops.git" "$OPS_BRANCH"
  echo "Successfully pushed GitOps updates"
fi
'''
              }
            }
          }
        }
      }
    }
  }

  post {
    always { cleanWs() }
    success {
      echo "🎉 Pipeline completed successfully!"
      echo "Image: ${imgRegistry}/${Namespace}/${PROJECT_NAME}:${params.TAG}"
    }
    failure {
      echo "❌ Pipeline failed. Check the logs above for details."
    }
  }
}
