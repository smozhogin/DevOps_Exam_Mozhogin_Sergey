terraform {
    required_version = ">= 1.3.0"

    required_providers {
        yandex = {
            source  = "yandex-cloud/yandex"
            version = "~> 0.130"
        }
    }
}

provider "yandex" {
    folder_id = var.folder_id
}

resource "yandex_serverless_container" "ml_service_api" {
    name               = "ml-service-api"
    folder_id          = var.folder_id
    service_account_id = var.service_account_id
    cores              = 1
    memory             = 256
    concurrency        = 1

    provision_policy {
        min_instances = 1
    }

    image {
        url = var.api_image_url
    }
}