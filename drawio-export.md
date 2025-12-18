# Draw.io Export Server – OpenShift Deployment Manual

This guide explains how to deploy `jgraph/export-server` (server for generating images/PDFs from draw.io format) into an OpenShift project for internal use by other services.

## Prerequisites

* Access to an OpenShift cluster

* Logged in with oc login

* Target project selected:

`oc project <your-project>`

1. Deploy the Export Server

* Create a file called jgraph-export-server-deployment.yaml:

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jgraph-export-server
  labels:
    app: jgraph-export-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jgraph-export-server
  template:
    metadata:
      labels:
        app: jgraph-export-server
    spec:
      containers:
        - name: export-server
          image: jgraph/export-server:latest
          ports:
            - containerPort: 8000
              protocol: TCP
          env:
            - name: EXPORT_SERVER_PORT
              value: "8000"
          resources:
            requests:
              memory: "1Gi"
            limits:
              memory: "2Gi"
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
```

* Apply the deployment:

`oc apply -f jgraph-export-server-deployment.yaml`

* Verify the pod is running:

`oc get pods -l app=jgraph-export-server`

2. Create an Internal Service

* Create a file called jgraph-export-server-service.yaml:

```
apiVersion: v1
kind: Service
metadata:
  name: jgraph-export-server
spec:
  type: ClusterIP
  selector:
    app: jgraph-export-server
  ports:
    - name: http
      protocol: TCP
      port: 8000
      targetPort: 8000
```

* Apply the service:

`oc apply -f jgraph-export-server-service.yaml`

## Troubleshooting

* Check logs:

`oc logs deployment/jgraph-export-server`

* Check pod status:

`oc describe pod -l app=jgraph-export-server`