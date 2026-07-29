import requests

r = requests.post("http://localhost:8000/predict/batch", json={
    "compounds": [
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "name": "Caffeine"}
    ]
})

if r.status_code != 200:
    print("ERROR:", r.status_code, r.text[:200])
else:
    data = r.json()
    for item in data["items"]:
        if item.get("error"):
            print(item["name"], ": ERROR -", item["error"])
        else:
            print(item["name"], ":", item["n_predicted"], "metabolites in", item["elapsed_s"], "s")
    print("Total:", data["completed"], "/", data["total"], "succeeded")
