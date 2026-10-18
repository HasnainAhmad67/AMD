import { Link, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell";
import AlertsPage from "./pages/AlertsPage";
import AssessPage from "./pages/AssessPage";
import AssessmentPage from "./pages/AssessmentPage";
import FleetPage from "./pages/FleetPage";
import MachinePage from "./pages/MachinePage";
import SettingsPage from "./pages/SettingsPage";

const NotFound = () => <div className="card"><b>Page not found</b><Link to="/" style={{ color: "var(--ac)" }}>Back to the fleet</Link></div>;
export default function App() {
  return (<Routes><Route element={<Shell />}>
    <Route index element={<FleetPage />} /><Route path="machines/:id" element={<MachinePage />} /><Route path="alerts" element={<AlertsPage />} />
    <Route path="assess" element={<AssessPage />} /><Route path="assessments/:id" element={<AssessmentPage />} /><Route path="settings" element={<SettingsPage />} />
    <Route path="*" element={<NotFound />} /></Route></Routes>);
}
