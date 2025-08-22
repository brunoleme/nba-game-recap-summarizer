terraform {
  backend "s3" {
    bucket         = "cypher2text-terraform-state-bucket"
    key            = "staging/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
