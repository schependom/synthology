package org.synthology.udm;

import java.io.FileOutputStream;
import java.io.OutputStream;

import org.apache.jena.rdf.model.InfModel;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.riot.Lang;
import org.apache.jena.riot.RDFDataMgr;

public final class JenaMaterializerCli {

    private JenaMaterializerCli() {
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: java -jar jena-materializer.jar <ontologyPath> <baseTriplesPath> <outputPath>");
            System.exit(1);
        }

        String ontologyPath = args[0];
        String baseTriplesPath = args[1];
        String outputPath = args[2];

        Model schemaModel = ModelFactory.createDefaultModel();
        RDFDataMgr.read(schemaModel, ontologyPath);

        Model baseModel = ModelFactory.createDefaultModel();
        RDFDataMgr.read(baseModel, baseTriplesPath);

        Reasoner reasoner = ReasonerRegistry.getOWLReasoner().bindSchema(schemaModel);
        InfModel infModel = ModelFactory.createInfModel(reasoner, baseModel);
        infModel.prepare();

        Model closure = ModelFactory.createDefaultModel();
        // Export the fully entailed graph from the InfModel. Some Jena reasoners
        // expose incomplete/empty deduction models for certain OWL features.
        closure.add(infModel);

        try (OutputStream output = new FileOutputStream(outputPath)) {
            RDFDataMgr.write(output, closure, Lang.NTRIPLES);
        } catch (Exception exception) {
            System.err.println("Failed to write materialized closure: " + exception.getMessage());
            exception.printStackTrace(System.err);
            System.exit(2);
        }
    }
}
