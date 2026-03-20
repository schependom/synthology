import java.io.FileOutputStream;

import org.apache.jena.ontology.OntModel;
import org.apache.jena.ontology.OntModelSpec;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.riot.RDFDataMgr;

public class JenaMaterializer {
    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage: java JenaMaterializer <ontology.ttl> <base_facts.ttl> <output.ttl>");
            return;
        }

        String ontologyPath = args[0];
        String dataPath = args[1];
        String outputPath = args[2];

        System.out.println("1. Setting up OWL Ontology Model...");
        OntModel ontModel = ModelFactory.createOntologyModel(OntModelSpec.OWL_MEM_RULE_INF);

        System.out.println("2. Loading TBox (Ontology)...");
        RDFDataMgr.read(ontModel, ontologyPath);

        System.out.println("3. Loading ABox (Base Facts) and Materializing...");
        RDFDataMgr.read(ontModel, dataPath);

        ontModel.prepare();

        System.out.println("4. Writing Materialized Graph to file...");
        try {
            FileOutputStream out = new FileOutputStream(outputPath);
            ontModel.write(out, "TURTLE");
            out.close();

            long totalSize = ontModel.size();
            System.out.println("Success! Total facts in materialized graph: " + totalSize);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
