# SERVICE.md

ABotClaw shared service registry.

All services in this file are assumed to be reachable by both OpenClaw and the robots.
Update this file first when the service host or port changes.

## Shared Service Host

- `SERVICE_HOST=<SERVICE_HOST>`

## Service Table

| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Robot memory write / query / retrieval | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health`, `/query/*`, `/memory/*` |
| YOLO | Object detection service | `<SERVICE_HOST>` | `8013` | `http://<SERVICE_HOST>:8013` | `/health`, `/detect` |
| VLAC | Task progress / completion critic | `<SERVICE_HOST>` | `8014` | `http://<SERVICE_HOST>:8014` | `/health`, `/critic` |
| GraspAnything | Grasp proposal / grasp detection service | `<SERVICE_HOST>` | `8015` | `http://<SERVICE_HOST>:8015` | `/health`, `/grasp/detect` |

## Notes

- Use `base64` for image transfer unless a service explicitly documents another format.
- For long flows, call `/health` first.
- For full request/response details, check each service's own API / agent docs.
