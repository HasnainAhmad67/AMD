import { Link, useParams } from "react-router-dom";
import { useAssessment, useMachines } from "../api/hooks";
import AssessmentView from "../components/AssessmentView";
import { ErrorCard, Skeletons } from "../components/QueryState";

export default function AssessmentPage() {
  const { id } = useParams(); const q = useAssessment(id); const machines = useMachines();
  if (q.isError) return <><Link to="/assess" className="sub">← New assessment</Link><ErrorCard error={q.error} onRetry={() => q.refetch()} title="Could not load this assessment" /></>;
  if (!q.data) return <Skeletons heights={[70, 60, 340, 300]} />;
  return <><Link to="/assess" className="sub">← New assessment</Link><AssessmentView a={q.data} name={machines.data?.find((m) => m.id === q.data.machine_id)?.name} /></>;
}
