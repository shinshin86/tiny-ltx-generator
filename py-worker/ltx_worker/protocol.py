def response(request_id, response_type, payload):
    return {"id": request_id, "type": response_type, "payload": payload}
