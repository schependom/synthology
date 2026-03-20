import java.io.FileOutputStream;
import org.apache.jena.rdf.model.*;
import org.apache.jena.reasoner.Reasoner;
import org.apache.jena.reasoner.ReasonerRegistry;
import org.apache.jena.riot.RDFDataMgr;

public class JenaMaterializer {
    public static void main(String[] args) {
        if (args.length < 3) return;

        Model tbox = RDFDataMgr.loadModel(args[0]);
        Model abox = RDFDataMgr.loadModel(args[1]);

        // Use the MICRO rule reasoner to avoid bnode explosion
        Reasoner reasoner = ReasonerRegistry.getOWLMicroReasoner();
        InfModel infModel = ModelFactory.createInfModel(reasoner, tbox, abox);

        // Force reasoning
        infModel.prepare();

        try {
            FileOutputStream out = new FileOutputStream(args[2]);
            // CRITICAL FIX: Only extract and write the NEWLY INFERRED facts
            Model deductions = infModel.getDeductionsModel();
            deductions.write(out, "TURTLE");
            out.close();

            System.out.println("Success! Total NOVEL facts inferred: " + deductions.size());
        } catch (Exception e) { e.printStackTrace(); }
    }
}
