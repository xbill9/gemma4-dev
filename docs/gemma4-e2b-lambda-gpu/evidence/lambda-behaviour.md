# Lambda behaviour seen on 2026-11-18/19 (captured from command output in this session)

## A second instance with only a minimum set

Scaling was `MinExecutionEnvironments` 1 with no maximum during the first-light benchmark (48 requests at
once at about 23:05 UTC). CloudWatch `AWS/Usage` ResourceCount, EC2 vCPU, Class G/OnDemand, us-east-1,
maximum per 5 minutes (local time, UTC-5): 17:58 8.0; 18:03 to 18:33 16.0; 18:38 8.0.
A second execution environment's log stream (`…77191f61`) starts at 23:06:46 UTC and shuts down at 23:09:28 UTC.

## Republish at the capacity provider's vCPU limit

- First republish with the drafter image, published 23:52:11 UTC: `LastUpdateStatus` Failed after 1,099.8 s,
  reason "An internal error occurred. Please try again."
- Second (`lmi.py function`), 00:15 UTC: Failed, "MinimumExecutionEnvironments can't be provisioned because
  CapacityProvider has reached maxVCpuCount limit. Increase maxVCpuCount in the CapacityProvider configuration
  and try again."
- After it, `$LATEST.PUBLISHED` held CodeSha256 b50a0edc… (the image without the drafter) and VLLM_SERVER_ARGS
  without `--speculative-config`, while `$LATEST` held ccdad352… with it. Environments started then logged
  `speculative_config=None` (see nodrafter-repeat-log-streams.txt).

## republish.py (scale to 0, publish, scale to 1/1), 2026-11-19

    19:38:35 scaled to 0: Deactivated Successful after 50 s
    published $LATEST.PUBLISHED ccdad352ab1c26e45835f8e3ba867f6cc95ab6ee8bbe104e94f5735cfec98cd2
    19:38:53 published code in place: Deactivated Successful after 17 s
    19:48:03 active: Active Successful after 548 s
    {"publish_to_code_s": 18.2, "scale_up_to_active_s": 552.0}
    platform.initReport durationMs 205420.123

## Image format

CreateFunction with the image Docker 29 pushed by default (manifest media type
application/vnd.oci.image.index.v1+json): InvalidParameterValueException "The image manifest, config or layer
media type for the source image … is not supported." The buildx image (application/vnd.docker.distribution.manifest.v2+json)
was accepted. Image push times: 398.4 s first push (base layers included), 19.6 s for the rebuild adding the drafter.
