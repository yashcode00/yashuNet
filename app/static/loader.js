const urlParams = new URLSearchParams(window.location.search);
const user_uuid = urlParams.get('user_uuid');
console.log(user_uuid);

function ajax_call() {
    let fileStatusList;
    const xhr = new XMLHttpRequest();
    const url = `/status/${user_uuid}`;
    xhr.open('GET', url, false);

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            fileStatusList = this.responseText;
        } else {
            return undefined;
        }
    }
    xhr.send();
    console.log(`Response from status/<id> -> ${JSON.parse(fileStatusList)}`);
    return JSON.parse(fileStatusList);
}

window.addEventListener('load', function (e) {
    console.log(user_uuid);
    if (!user_uuid) {
        window.location.replace("/")
    }
    const interval = this.window.setInterval(() => {
        fileStatus = ajax_call();
        let flag = false;
        for (let i = 0; i < fileStatus.length; i++) {
            if (fileStatus[i]?.status === "Pending") {
                flag = true;
                break;
            }
        }

        if (!flag) {
            // All files are processed
            this.window.clearInterval(interval);
            window.location.replace(`/display?user_uuid=${user_uuid}`);
        }
    }, 500);
});