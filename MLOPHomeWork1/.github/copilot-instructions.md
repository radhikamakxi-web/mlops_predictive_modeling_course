# Project instructions

- [x] Verify workspace instructions exist.
- [x] Clarify requirements: implement the supplied Fashion-MNIST MLOps assignment.
- [ ] Scaffold the Python project in this workspace.
- [ ] Implement training, tracking, storage, API, and UI.
- [x] Install required extensions: none required.
- [ ] Validate the project and run tests.
- [ ] Create a test task.
- [ ] Launch and verify services when Docker is available.
- [ ] Complete setup and operational documentation.

Use Python 3.11 or newer and a src-layout package named fashion_mnist.
Keep training separate from inference. Never commit credentials or downloaded artifacts.
Preserve the required bucket spelling fashion-minst-bucket and version models by MLflow run ID.
Send completion notifications only after successful MLflow finalization and verified MinIO upload.
Use shared preprocessing for training and inference. Cover error paths with tests.