# safetensors-exporter
Export PyTorch models to Safetensors

## Summary

Export T5 models from PyTorch to Safetensors format.

## Usage

```shell

docker run -rm -v $(pwd)/models:/exported_model milarze/safetensors-exporter:latest \
    <Salesforce/codet5-small>
```

## Build

```shell
docker build -t milarze/safetensors-exporter:latest .
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
