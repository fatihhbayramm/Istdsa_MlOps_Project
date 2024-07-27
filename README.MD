# E-Posta Spam Sınıflandırma Modeli

Bu proje, e-posta spam sınıflandırması için geliştirilmiş bir modelin dağıtımını ve yönetimini kapsamaktadır. Model, FastAPI ile sunulmakta ve kullanıcı arayüzü Streamlit ile oluşturulmuştur. Docker kullanarak her iki bileşen konteynerize edilmiş ve Docker Compose ile birleştirilmiştir.

## Proje Hedefi

Bu projede, e-posta spam sınıflandırması için bir model geliştirdik ve bu modelin dağıtımını gerçekleştirdik. Modelimiz, FastAPI ile sunulmakta olup, Streamlit ile kullanıcı dostu bir arayüz sağlanmaktadır. Docker konteyner teknolojisi kullanılarak, uygulamanın her iki bileşeni de kapsayıcı içinde çalıştırılmakta ve Docker Compose ile yönetilmektedir.

## Model Seçimi

Spam sınıflandırma modelimizde **Precision** metriğini önceliklendirdik çünkü iş gereksinimlerimiz doğrultusunda yanlış pozitiflerin en aza indirilmesi kritik öneme sahiptir. Naive Bayes modelimiz, en yüksek Precision sonucunu vererek seçilmiştir ve FastAPI ile entegre edilmiştir.

## Backend Yapılandırması

- **Model Yerleştirme:** `spam_classifier_pipeline.pkl` dosyasını uygun klasöre yerleştirdik.
- **Sunucu Oluşturma:** `server.py` dosyasını oluşturduk ve FastAPI kullanarak API'yi tanımladık.
- **Gerekli Paketler:** `requirements.txt` dosyasına gerekli Python paketlerini ekledik.
- **Docker Container:** Dockerfile oluşturarak arka yüz uygulamasını konteynerize ettik.

## Frontend Yapılandırması

- **Streamlit ile Arayüz:** `app.py` dosyasını oluşturduk ve FastAPI API'sine veri gönderen bir arayüz geliştirdik.
- **Gerekli Paketler:** Frontend için gerekli Python paketlerini `requirements.txt` dosyasına ekledik.
- **Docker Container:** Frontend için Dockerfile oluşturarak konteynerize ettik.

## CI/CD Süreci

- **GitHub Actions:** CI/CD sürecini GitHub Actions ile otomatikleştirdik. `.github/workflows/cicd.yml` dosyasını oluşturup, build ve deploy işlemlerini yapılandırdık.
- **Secrets Oluşturma:** GitHub'da `CREDENTIALS_JSON`, `SERVICE_ACCOUNT`, ve `PROJECT_ID` gibi secrets'lar oluşturarak güvenli veri yönetimi sağladık.
- **Google Cloud Entegrasyonu:** Docker imajımızı Google Cloud Artifact Registry'ye yükledik, Google Cloud Run ile dağıttık ve HTTPS üzerinden erişimi test ettik.

## Son Kontroller

- **Docker İmajı:** Google Cloud Console üzerinden Docker imajımızın Artifact Registry'de başarıyla depolandığını kontrol ettik.
- **Cloud Run:** Google Cloud Run'da imajın başarıyla çalıştığını doğruladık.
- **API Erişimi:** FastAPI arayüzüne verilen HTTPS URL üzerinden erişimi test ettik.

Bu adımlar, modelimizin başarılı bir şekilde dağıtıldığını ve sistemin beklendiği gibi çalıştığını doğrulamanızı sağlar.

## Kullanım

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Depoyu Klonlayın:**
   ```bash
   git clone https://github.com/fatihhbayramm/Istdsa_MlOps_Project.git
2. **Gerekli Paketleri Kurun:**
   ```bash
   pip install -r requirements.txt
3. **Docker Compose ile Çalıştırın:**
   ```bash
   docker-compose up --build

4. **Frontend ve Backend'e Erişin:**
 
    Frontend URL: http://localhost:8501
    Backend URL: http://localhost:8000
