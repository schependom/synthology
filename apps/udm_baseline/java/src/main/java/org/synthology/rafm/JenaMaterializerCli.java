package org.synthology.udm;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.File;
import java.util.Iterator;

import org.apache.jena.graph.Triple;
import org.apache.jena.rdf.model.InfModel;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.ResourceFactory;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.reasoner.Derivation;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.reasoner.rulesys.RuleDerivation;
import org.apache.jena.riot.Lang;
import org.apache.jena.riot.RDFDataMgr;

public final class JenaMaterializerCli {

    private JenaMaterializerCli() {
    }

    public static void main(String[] args) {
        if (args.length != 3 && args.length != 4) {
            System.err.println(
                    "Usage: java -jar jena-materializer.jar <ontologyPath> <baseTriplesPath> <outputPath> [owl_micro|owl_mini|owl_full]");
            System.exit(1);
        }

        String ontologyPath = args[0];
        String baseTriplesPath = args[1];
        String outputPath = args[2];
        String profile = args.length == 4 ? args[3] : "owl_mini";

        Model schemaModel = ModelFactory.createDefaultModel();
        RDFDataMgr.read(schemaModel, ontologyPath);

        Model baseModel = ModelFactory.createDefaultModel();
        RDFDataMgr.read(baseModel, baseTriplesPath);

        Reasoner reasoner = buildReasoner(profile).bindSchema(schemaModel);
        reasoner.setDerivationLogging(true);
        InfModel infModel = ModelFactory.createInfModel(reasoner, baseModel);
        infModel.prepare();

        Model closure = ModelFactory.createDefaultModel();
        closure.add(infModel);

        File outDir = new File(outputPath).getParentFile();
        String hopsOutputPath = outDir == null ? "hops.csv" : new File(outDir, "hops.csv").getPath();

        try (OutputStream output = new FileOutputStream(outputPath);
             PrintWriter hopsOut = new PrintWriter(new FileOutputStream(hopsOutputPath))) {
             
            RDFDataMgr.write(output, closure, Lang.NTRIPLES);
            
            hopsOut.println("subject,predicate,object,depth");
            for (Statement stmt : closure.listStatements().toList()) {
                int depth = calculateDepth(infModel, stmt);
                if (depth > 0) {
                    hopsOut.println(String.format("%s,%s,%s,%d", 
                        stmt.getSubject().toString(),
                        stmt.getPredicate().toString(),
                        stmt.getObject().toString(),
                        depth
                    ));
                }
            }
        } catch (Exception exception) {
            System.err.println("Failed to write materialized closure: " + exception.getMessage());
            exception.printStackTrace(System.err);
            System.exit(2);
        }
    }

    public static int calculateDepth(InfModel infModel, Statement stmt) {
        Iterator<Derivation> derivations = infModel.getDerivation(stmt);
        if (derivations == null || !derivations.hasNext()) {
            return 0; // Base fact
        }
        
        int maxParentDepth = 0;
        Derivation deriv = derivations.next(); // Just take the first proof
        
        if (deriv instanceof RuleDerivation) {
            RuleDerivation ruleDeriv = (RuleDerivation) deriv;
            for (Triple premise : ruleDeriv.getMatches()) {
                Statement premiseStmt = infModel.asStatement(premise);
                maxParentDepth = Math.max(maxParentDepth, calculateDepth(infModel, premiseStmt));
            }
        }
        return maxParentDepth + 1;
    }

    private static Reasoner buildReasoner(String profile) {
        String normalized = profile == null ? "owl_mini" : profile.trim().toLowerCase();
        switch (normalized) {
            case "owl_micro":
                return ReasonerRegistry.getOWLMicroReasoner();
            case "owl_full":
                return ReasonerRegistry.getOWLReasoner();
            case "owl_mini":
            default:
                return ReasonerRegistry.getOWLMiniReasoner();
        }
    }
}
