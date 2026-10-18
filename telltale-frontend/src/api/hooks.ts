import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

// Poll at roughly the simulator tick rate so the fleet feels live.
export const LIVE_MS = 3000;

export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    // Check often while the first-start model training runs, then relax.
    refetchInterval: (q) => (q.state.data?.sensor_bootstrap?.status === "running" || q.state.status === "error" ? 2000 : 15000),
  });
export const useOverview = () => useQuery({ queryKey: ["overview"], queryFn: api.overview, refetchInterval: LIVE_MS });
export const useMachines = () => useQuery({ queryKey: ["machines"], queryFn: api.machines, refetchInterval: LIVE_MS });
export const useMachine = (id: string) =>
  useQuery({ queryKey: ["machine", id], queryFn: () => api.machine(id), refetchInterval: LIVE_MS });
export const useAlerts = (status?: string) =>
  useQuery({ queryKey: ["alerts", status ?? "all"], queryFn: () => api.alerts(status), refetchInterval: LIVE_MS });
export const useSensorSchema = (enabled = true) =>
  useQuery({ queryKey: ["sensor-schema"], queryFn: api.sensorSchema, enabled, staleTime: Infinity });
export const useAcousticModels = () => useQuery({ queryKey: ["acoustic-models"], queryFn: api.acousticModels });
export const useSamples = () => useQuery({ queryKey: ["samples"], queryFn: api.samples });
export const useModelStatus = () => useQuery({ queryKey: ["models"], queryFn: api.models });

export const useAssessment = (id?: string) =>
  useQuery({ queryKey: ["assessment", id], queryFn: () => api.assessment(id as string), enabled: Boolean(id), retry: false });
export const useAssessments = (limit = 8) =>
  useQuery({ queryKey: ["assessments", limit], queryFn: () => api.assessments(limit), retry: false });
