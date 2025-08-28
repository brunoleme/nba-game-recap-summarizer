terraform {
  backend "s3" {
    bucket         = "nba-recap-model-tf-state-bucket"
    key            = "staging/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
