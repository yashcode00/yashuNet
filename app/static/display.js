const urlParams = new URLSearchParams(window.location.search);
const user_uuid = urlParams.get('user_uuid');

// selecting required element
const paginationEle = document.querySelector(".pagination ul");
const displayEle = document.querySelector('.showcase');

var globalData;
let page = 1;

function ajax_call() {
    let fileStatusList;
    const xhr = new XMLHttpRequest();
    const url = `/getdata/${user_uuid}`;
    xhr.open('GET', url, false);

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            fileStatusList = this.responseText;
        } else {
            return undefined;
        }
    }
    xhr.send();
    return JSON.parse(fileStatusList);
}

//calling function with passing parameters and adding inside element which is ul tag
function createDisplay(page){
    let totalPages = globalData.length;
    let liTag = '';
    let imgTag = '';
    let active;
    let beforePage = page - 1;
    let afterPage = page + 1;
    paginationEle.classList.remove('d-none');
    
    imgTag += `<div><img src=/${globalData[page-1].path} class="result-img" />`;
    imgTag += `<img src=/${globalData[page-1].path} class="result-img" /></div>`;

    displayEle.innerHTML = imgTag;
    console.log(totalPages);
    console.log(displayEle);

    if (totalPages==1) {
        paginationEle.classList.add('d-none');
        paginationEle.innerHTML = liTag;
        return;
    }
    
    if(page > 1){ //show the next button if the page value is greater than 1
        liTag += `<li class="btn prev" onclick="createPagination(${page - 1})"><span><i class="fas fa-angle-left"></i> Prev</span></li>`;
    }

    //   if(page > 2){ //if page value is less than 2 then add 1 after the previous button
    //     liTag += `<li class="first numb" onclick="createPagination(totalPages, 1)"><span>1</span></li>`;
    //     if(page > 3){ //if page value is greater than 3 then add this (...) after the first li or page
    //       liTag += `<li class="dots"><span>...</span></li>`;
    //     }
    //   }

    //how many pages or li show before the current li
    if (page == totalPages) {
        beforePage = beforePage - 1;
    } else if (page == totalPages - 1) {
        beforePage = beforePage;
    }
    //   beforePage = min(1, page-1);

    //   // how many pages or li show after the current li
    if (page == 1) {
        afterPage = afterPage + 1;
    } else if (page == 2) {
        afterPage  = afterPage;
    }
    // afterPage = max(totalPages, page+1);

    for (var plength = beforePage; plength <= afterPage; plength++) {
        if (plength > totalPages) { //if plength is greater than totalPage length then continue
        continue;
        }
        if (plength == 0) { //if plength is 0 than add +1 in plength value
        plength = plength + 1;
        }
        if(page == plength){ //if page is equal to plength than assign active string in the active variable
        active = "active";
        }else{ //else leave empty to the active variable
        active = "";
        }
        liTag += `<li class="numb ${active}" onclick="createPagination(${plength})"><span>${plength}</span></li>`;
    }

    //   if(page < totalPages - 1){ //if page value is less than totalPage value by -1 then show the last li or page
    //     if(page < totalPages - 2){ //if page value is less than totalPage value by -2 then add this (...) before the last li or page
    //       liTag += `<li class="dots"><span>...</span></li>`;
    //     }
    //     liTag += `<li class="last numb" onclick="createPagination(totalPages, ${totalPages})"><span>${totalPages}</span></li>`;
    //   }

    if (page < totalPages) { //show the next button if the page value is less than totalPage(20)
        liTag += `<li class="btn next" onclick="createPagination(${page + 1})"><span>Next <i class="fas fa-angle-right"></i></span></li>`;
    }

    paginationEle.innerHTML = liTag; //add li tag inside ul tag

}

window.addEventListener('load', function (e) {
    console.log(user_uuid);
    if (!user_uuid) {
        window.location.replace("/");
    }
    globalData = ajax_call();
    if (!globalData || globalData.length==0) {
        window.location.replace("/");
    }
    createDisplay(1);
});

