# SIP endpoint mapping for each intent

SIP_ENDPOINTS = {
    "SALES": "sip:sales@company.com",
    "BILLING": "sip:billing@company.com",
    "SUPPORT": "sip:support@company.com",
    "TECHNICAL": "sip:technical@company.com",
    "HR": "sip:hr@company.com",
    "OPERATIONS": "sip:operations@company.com",
    "EMERGENCY": "sip:emergency@company.com",
    "GENERAL": "sip:reception@company.com",
}


def route_intent(intent):
    """
    Convert a predicted intent into its SIP endpoint.
    """

    intent = intent.upper()

    endpoint = SIP_ENDPOINTS.get(intent)

    if endpoint is None:
        return "sip:reception@company.com"

    return endpoint


if __name__ == "__main__":

    test_intents = [
        "SALES",
        "BILLING",
        "SUPPORT",
        "TECHNICAL",
        "HR",
        "OPERATIONS",
        "EMERGENCY",
        "GENERAL",
    ]

    print("SIP Intent Router Test")
    print("-" * 40)

    for intent in test_intents:
        endpoint = route_intent(intent)

        print(f"Intent: {intent}")
        print(f"SIP Endpoint: {endpoint}")
        print("-" * 40)