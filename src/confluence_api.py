import logging
import requests

logger = logging.getLogger(__name__)


# https://developer.atlassian.com/server/confluence/confluence-rest-api-examples/
class Confluence:
    def __init__(self, access_token: str):
        self._token = access_token

    def _request(self, method: str, endpoint: str, data = None, params = None) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self._token
        }
        response = requests.request(method=method, url=f"https://confluence.cern.ch/rest/api/{endpoint}", headers=headers, json=data, params=params)
        logger.info("%s %s: status %d", method, endpoint, response.status_code)
        # logger.info("%s %s: response %s", method, endpoint, response.content.decode(errors="ignore"))

        if 400 <= response.status_code < 500:
            try:
                message = response.json()["message"]
            except Exception:
                logger.info("body %s", response.content.decode(errors="ignore"))
            else:
                raise Exception("API error: " + message)
        response.raise_for_status()

        return response

    def _get(self, endpoint: str, **kwargs) -> dict:
        return self._request("GET", endpoint, **kwargs).json()

    def _post(self, endpoint: str, data: dict) -> dict:
        return self._request("POST", endpoint, data).json()

    def _put(self, endpoint: str, data: dict) -> dict:
        return self._request("PUT", endpoint, data).json()

    def insert_or_update_page(self, space_key: str, ancestor_id: str, title: str, content: str):
        """
        Insert a Confluence page, or update it if it already exists (matched by title)

        Ancestor ID is the numeric ID of the parent page.

        Content must be in Confluence Storage Format: https://confluence.atlassian.com/display/DOC/Confluence+Storage+Format
        """
        response = self._get("content", params={"title": title, "spaceKey": space_key, "expand": "version"})

        if len(response["results"]) > 0:
            (result,) = response["results"]
            id = result["id"]
            version = result["version"]["number"]

            data = {
                "id": id,
                "type": "page",
                "title": title,
                "space": {"key": space_key},
                "body": {
                    "storage": {
                        "value": content,
                        "representation":"storage"
                    }
                },
                "version" : {
                    "number": version + 1,
                    "minorEdit" : True  # supress notification to watchers
                }
            }

            response = self._put(f"content/{id}", data)
        else:
            data = {
                "type": "page",
                "title": title,
                "ancestors": [{"id": ancestor_id}],
                "space": {"key": space_key},
                "body": {
                    "storage": {
                        "value": content,
                        "representation": "storage"
                    }
                }
            }

            self._post("content", data)
