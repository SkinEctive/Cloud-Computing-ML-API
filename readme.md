# Machine Learning API Documentation 🧑‍💻

## SkinEctive Machine Learning API 🔗

```http
  http://localhost:5000/
```
## Machine Learning API Endpoints

| Endpoint                 | Method | Input        | Description                             | Status      |
| ------------------------ | ------ | ------------ | --------------------------------------- | ----------- |
| /detect                  | POST   | Image (file) | Create a new detection                  | ✅ Completed |
| /detect/history          | GET    | -            | Get all detection history               | ✅ Completed |
| /detect/history/<userId> | GET    | -            | Get detection history for specific user | ✅ Completed |