# P2 - provenance for Methods §2.8

## Sentence to paste into Methods

Sequences were retrieved from UniProtKB (UniProt release 2026_03 (released 02-September-2026)), accessed 25 September 2026. Analyses were run under Windows 11 (AMD64) with Python 3.12.10. Protein language model embeddings used facebook/esm2_t33_650M_UR50D. Package versions are listed in Supplementary Table S7, and all random seeds were fixed at 20260925.

## Supplementary Table S7 - software versions

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| pandas | 2.3.3 |
| biopython | 1.87 |
| requests | 2.34.2 |
| transformers | 5.17.0 |
| torch | 2.12.0 |
| matplotlib | 3.10.8 |
| networkx | 3.6.1 |
| openpyxl | 3.1.5 |
| huggingface-hub | 1.32.0 |
| safetensors | 0.8.0 |
| Python | 3.12.10 |
| OS | Windows 11 (AMD64) |

## UniProt response headers (verbatim)

```
Vary: accept,accept-encoding,x-uniprot-release,x-api-deployment-date, User-Agent, Accept-Encoding
Cache-Control: public, max-age=43200
x-cache: hit cached
Content-Type: text/plain;format=tsv
Content-Encoding: gzip
Access-Control-Allow-Credentials: true
Access-Control-Expose-Headers: Link, X-Total-Results, X-UniProt-Release, X-UniProt-Release-Date, X-API-Deployment-Date
X-API-Deployment-Date: 23-September-2026
strict-transport-security: max-age=31536000; includeSubDomains; preload
Date: Fri, 25 Sep 2026 19:39:57 GMT
Access-Control-Max-Age: 1728000
X-UniProt-Release: 2026_03
X-Total-Results: 1
Access-Control-Allow-Origin: *
Connection: keep-alive
Access-Control-Allow-Methods: GET, PUT, POST, DELETE, PATCH, OPTIONS
Access-Control-Allow-Headers: DNT,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization
Content-Length: 39
X-UniProt-Release-Date: 02-September-2026
```
