output "alb_url" {
  value = "http://${module.alb.alb_dns_name}"
}