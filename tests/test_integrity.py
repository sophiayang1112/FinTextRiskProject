import io
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit
from fintext_risk.evidence import EvidenceStore,clean_text,mine
from fintext_risk.data import simulate,chronological_partitions
from fintext_risk.modeling import model,integrated_gradients,make_features,attach_llm,fingerprint
from fintext_risk.governance import fit_temperature,conformal_cutoff,prediction_sets,psi
from fintext_risk.llm import assess


def documents():
    return pd.DataFrame([
        dict(doc_id="past",entity_id="E",published_at="2024-01-01",available_at="2024-01-02",quality=.9,text="late payment and liquidity shortfall"),
        dict(doc_id="delayed",entity_id="E",published_at="2024-01-01",available_at="2024-04-01",quality=1.,text="late payment under review"),
        dict(doc_id="future",entity_id="E",published_at="2024-03-01",available_at="2024-03-02",quality=1.,text="late payment in future"),
        dict(doc_id="other",entity_id="OTHER",published_at="2024-01-01",available_at="2024-01-02",quality=1.,text="late payment other entity")])


class EvidenceTests(unittest.TestCase):
    def test_asof_and_entity_filter(self):
        store=EvidenceStore(documents(),"2024-02-01")
        found=store.retrieve("E","2024-02-01","late payment")
        self.assertEqual([row["doc_id"] for row in found],["past"])
        self.assertEqual(store.retrieve("UNKNOWN","2024-02-01","late payment"),[])

    def test_duplicate_content_retains_earliest_availability(self):
        docs=documents();duplicate=docs.iloc[[0]].copy();duplicate["doc_id"]="copy"
        duplicate["available_at"]="2024-01-05"
        store=EvidenceStore(pd.concat([docs,duplicate]),"2024-02-01")
        self.assertEqual(len(store.documents),4)

    def test_time_decay_and_quality_weight(self):
        store=EvidenceStore(documents(),"2024-02-01")
        found=store.retrieve("E","2024-02-01","late payment")[0]
        self.assertAlmostEqual(found["weight"],.9*np.exp(-.005*31))

    def test_retrieval_vocabulary_not_fit_on_future_documents(self):
        docs=documents();docs.loc[docs.doc_id=="future","text"]="uniquefuturetoken late payment"
        store=EvidenceStore(docs,"2024-02-01")
        self.assertNotIn("uniquefuturetoken",store.vectorizer.vocabulary_)
        self.assertEqual(store.retrieve("E","2024-02-01","unmatchedvocabulary"),[])

    def test_graph_has_document_event_provenance(self):
        graph=EvidenceStore(documents(),"2024-02-01").graph()
        self.assertTrue(any(edge["relation"]=="mentions" for edge in graph["edges"]))
        self.assertTrue(any(node["id"]=="event:payment_delay" for node in graph["nodes"]))

    def test_redaction_and_negation(self):
        cleaned=clean_text("Contact person@example.com or 202-555-0123")
        self.assertNotIn("person@example.com",cleaned)
        self.assertNotIn("202-555-0123",cleaned)
        self.assertEqual(mine("no late payment; stable revenue")["events"],[])


class GovernanceTests(unittest.TestCase):
    def test_label_maturity_purge(self):
        apps,_=simulate(600,19);parts=chronological_partitions(apps)
        for name,next_name in [("train","validation"),("validation","calibration"),("calibration","test")]:
            cutoff=apps.loc[parts[next_name],"assessed_at"].min()
            self.assertTrue((apps.loc[parts[name],"label_available_at"]<=cutoff).all())
        self.assertEqual(len(set().union(*[set(ids) for ids in parts.values()])),sum(map(len,parts.values())))

    def test_temperature_and_prediction_sets(self):
        logits=np.array([-2.,-.5,.5,2.]);labels=np.array([0,0,1,1])
        temperature=fit_temperature(logits,labels)
        self.assertGreater(temperature,0)
        cutoff=conformal_cutoff(np.array([.1,.2,.8,.9]),labels,alpha=.25)
        self.assertEqual(prediction_sets([.1,.9],cutoff),[[0],[1]])
        self.assertEqual(psi(np.arange(20),np.arange(20)),0)

    def test_integrated_gradient_completeness(self):
        x=pd.DataFrame({"x":[0.,1.,2.,3.,4.,5.],"z":[1.,0.,1.,0.,1.,0.]})
        estimator=model(["x","z"],text=False).fit(x,[0,0,0,1,1,1])
        predictions=estimator.predict_proba(x)[:,1]
        baseline=expit(estimator.named_steps["model"].intercept_[0])
        attribution=integrated_gradients(estimator,x)
        np.testing.assert_allclose([sum(row["attribution"] for row in record) for record in attribution],predictions-baseline,atol=1e-10)


class LocalLLMTests(unittest.TestCase):
    def test_external_endpoint_blocked(self):
        with self.assertRaises(ValueError):assess({},[{"doc_id":"D"}],"model","https://example.com")

    def test_mocked_response_validates_citations(self):
        body={"message":{"content":json.dumps({"risk_score":.4,"evidence_ids":["D"],"explanation":"Evidence supports review"})}}
        with patch("urllib.request.urlopen",return_value=io.BytesIO(json.dumps(body).encode())):
            result=assess({},[{"doc_id":"D","text":"data"}],"local-model")
        self.assertEqual(result["risk_score"],.4)
        body["message"]["content"]=json.dumps({"risk_score":.4,"evidence_ids":["invented"],"explanation":"bad"})
        with patch("urllib.request.urlopen",return_value=io.BytesIO(json.dumps(body).encode())):
            with self.assertRaises(ValueError):assess({},[{"doc_id":"D"}],"local-model")

    def test_signal_input_hash_rejected_on_mismatch(self):
        apps=pd.DataFrame([{ "sample_id":"S","entity_id":"E","assessed_at":"2024-02-01","narrative":"late payment"}])
        record={"sample_id":"S","input_hash":"wrong","risk_score":.5,"evidence_ids":["D"]}
        with self.assertRaises(ValueError):attach_llm(pd.DataFrame([{}]),apps,[record],[{"evidence":[{"doc_id":"D"}]}])


if __name__=="__main__":unittest.main()
